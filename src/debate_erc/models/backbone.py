"""LLaMA2-7B 主干加载（复现教训护栏：加载后校验 pad_token_id，缺失 raise RuntimeError）。"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.logging import get_logger

logger = get_logger("debate_erc.models")


def load_backbone(
    model_name: str = "NousResearch/Llama-2-7b-hf",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """加载主干模型与 tokenizer。

    护栏（设计报告八）：
    - pad_token_id 缺失时设置 pad_token = eos_token，兜底后仍缺失则 raise RuntimeError
      （不用 assert：-O 模式下 assert 会被剥除，护栏失效）
    - 优先 bf16（Blackwell 更稳定）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token 缺失，已设置 pad_token = eos_token (id=%s)", tokenizer.pad_token_id)
    if tokenizer.pad_token_id is None:
        # 红线护栏：用 RuntimeError 而非 assert（assert 可被 python -O 剥除后静默放行）
        raise RuntimeError(
            "pad_token_id 缺失（红线护栏）：pad_token=eos_token 兜底后仍为空，"
            "padding 会静默错位，拒绝加载该 tokenizer"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


class LLMGenerator:
    """统一生成后端：Agent 共用的采样生成 + 标签置信度打分。"""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def score_labels(self, prompt: str, labels: list[str] | None = None,
                     max_length: int = 2048) -> dict[str, float]:
        """候选标签的 teacher-forcing 序列后验概率（选择性辩论门控的校准信号）。

        返回 {label: prob}，prob 为候选集内 softmax(序列 logprob) 归一概率，
        argmax 即受限分类决策、top1 概率即校准置信度。

        关键修正（Layer 0 实证）：LLaMA2 SentencePiece 下 " label" 的首 token
        恒为公共空格符 29871，旧 label_confidence 取首 token 会让所有标签概率
        完全相等（7 标签退化为 1/7）。正确做法是对完整 " {label}" token 序列
        teacher-forcing 累加条件 logprob：LLaMA2 下公共空格项在候选间相同
        （softmax 中为常数偏移，不影响结果），Qwen 等 BPE 词表下 " neutral"
        可能是单个 ▁neutral token（无独立空格符），保留完整序列才能跨词表
        通用。与 SFT 答案 " {label}\\n" 的 token 口径一致。

        长度口径：使用序列 logsum（MLE 口径，与 SFT 逐 token CE 一致）。
        标签长度差异（1~3 子词）带来的偏差经 Layer 0 全量实证：该口径的 top1
        概率在纯 SFT 底座上分桶校准单调（<0.5 桶 acc 0.48 → ≥0.9 桶 acc 0.90），
        作为**门控信号**有效；但不应用它替代贪心生成做预测（logsum 对短标签
        有偏，W-F1 低于贪心生成口径）。
        """
        cand = labels or getattr(self, "candidate_labels", None)
        if not cand:
            raise ValueError("score_labels 需要 labels 参数或预设 candidate_labels")
        device = self.device
        # truncation=False：不让 tokenizer 右截断砍尾（保尾语义全权交给下方左截断）
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=False
        ).input_ids
        # 左截断保尾并重新前置 BOS（与 SFT left_truncate_with_bos 同语义）
        max_prompt = max_length - 4
        if len(prompt_ids) > max_prompt:
            bos = self.tokenizer.bos_token_id
            head = [bos] if bos is not None and bos != prompt_ids[0] else []
            prompt_ids = head + prompt_ids[-(max_prompt - len(head)):]

        label_tokens: dict[str, list[int]] = {}
        for lb in cand:
            # 完整 " {label}" 序列：不剥离首 token（跨词表通用：LLaMA2 首
            # token 为公共空格是常数偏移；Qwen BPE 可能整词单 token）
            ids = self.tokenizer.encode(f" {lb}", add_special_tokens=False)
            if not ids:
                raise ValueError(f"标签 {lb} 的编码为空")
            label_tokens[lb] = ids

        plen = len(prompt_ids)
        maxllen = max(len(v) for v in label_tokens.values())
        total_len = plen + maxllen
        bsz = len(cand)
        input_ids = torch.full(
            (bsz, total_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attn = torch.zeros((bsz, total_len), dtype=torch.long)
        tgt = torch.zeros((bsz, maxllen), dtype=torch.long)
        mask = torch.zeros((bsz, maxllen), dtype=torch.float32)
        for i, lb in enumerate(cand):
            toks = label_tokens[lb]
            seq = prompt_ids + toks
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
            tgt[i, :len(toks)] = torch.tensor(toks)
            mask[i, :len(toks)] = 1.0
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        logits = self.model(
            input_ids=input_ids, attention_mask=attn
        ).logits[:, plen - 1:plen + maxllen - 1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask
        log_sums = tok_lp.sum(dim=1)
        probs = torch.softmax(log_sums, dim=0).cpu().tolist()
        return {lb: float(p) for lb, p in zip(cand, probs)}

    @torch.no_grad()
    def label_confidence(self, prompt: str, label: str) -> float:
        """旧接口保留为 score_labels 的单标签包装。

        注意：旧实现取标签首 token（公共空格符）导致恒等退化，已废弃；
        现转发至 score_labels 的归一概率。
        """
        return self.score_labels(prompt).get(label, 0.5)
