"""DPO 训练器（V3 6.2 步骤 6 的后续训练）：手写 DPO 循环（不用 trl，规避闭包 OOM）。

核心约定（S3 契约）：
- policy logprob 用当前模型（含活跃 LoRA adapter）；
- reference logprob 用 `with model.disable_adapter():`（peft PeftModel 上下文管理器）；
  model 必须是 PeftModel——非 PeftModel 在 __init__ 直接 raise RuntimeError，
  禁止静默退化为 ref=policy（类 CPO）的失效参考信号；
- 参考前向以 model.eval() 包裹后恢复 train()：参考分布是纯推理口径，
  dropout 污染会让 β*(π-ref) 带入随机噪声；
- epoch 落盘仅保存 adapter（PeftModel.save_pretrained 默认只写 adapter 权重，
  7B 主干不落盘）；
- 序列 logprob 只对 response tokens（chosen/rejected 部分）求和；
  可选按 token 数归一（length_norm，默认 False）；
- loss = -logsigmoid(β*((π_c-ref_c)-(π_r-ref_r))) 的 batch 均值；
  显式记录 reward margins（β 缩放后的 chosen-rejected 差）与 chosen 排名正确率；
- 超长样本左截断 prompt 后重新前置 BOS；epoch 末 flush 残余梯度前做
  grad_accum/k 梯度补偿（与 SFTTrainer 同构，S5 契约）。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

from ..config import TrainingConfig
from ..data.schemas import PreferencePair
from ..utils.logging import get_logger
from .sft_trainer import _MAX_LEN, left_truncate_with_bos

_MAX_RESPONSE_TOKENS = 512  # 偏好对 response（" {label}" 短格式）的 token 上限


class _DPODataset(Dataset):
    """PreferencePair 列表 → 预编码（prompt / chosen / rejected token ids）。"""

    def __init__(self, pairs: list[PreferencePair], tokenizer, logger=None):
        bos_id = getattr(tokenizer, "bos_token_id", None)
        self.examples: list[dict] = []
        for p in pairs:
            # prompt 不做 tokenizer 右截断（R2-m4，与 SFTTrainer 同口径）：
            # truncation=True 会先砍掉 prompt 尾部，架空 left_truncate_with_bos
            # 的保尾语义；超长保护全权交给下方左截断
            prompt_ids = list(
                tokenizer(p.prompt, add_special_tokens=True, truncation=False).input_ids
            )
            chosen_ids = list(
                tokenizer(p.chosen, add_special_tokens=False, truncation=True,
                          max_length=_MAX_RESPONSE_TOKENS).input_ids
            )
            rejected_ids = list(
                tokenizer(p.rejected, add_special_tokens=False, truncation=True,
                          max_length=_MAX_RESPONSE_TOKENS).input_ids
            )
            # 总长保护：超长时左截断 prompt（保尾）并重新前置 BOS（chosen/rejected
            # response 长度不同 → 分别截断，保证两条序列各自 ≤ _MAX_LEN）
            prompt_ids_c, orig_c = left_truncate_with_bos(
                prompt_ids, len(chosen_ids), bos_id, _MAX_LEN
            )
            prompt_ids_r, orig_r = left_truncate_with_bos(
                prompt_ids, len(rejected_ids), bos_id, _MAX_LEN
            )
            if max(orig_c, orig_r) > _MAX_LEN and logger is not None:
                logger.warning(
                    "[dpo] 偏好对超长（prompt %d tokens，总长 %d > %d）："
                    "prompt 已左截断并重新前置 BOS",
                    len(prompt_ids), max(orig_c, orig_r), _MAX_LEN,
                )
            self.examples.append({
                "prompt_ids": prompt_ids,
                "chosen": {"prompt_ids": prompt_ids_c, "resp_ids": chosen_ids},
                "rejected": {"prompt_ids": prompt_ids_r, "resp_ids": rejected_ids},
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def _collate(batch: list[dict], pad_id: int) -> dict:
    """交错展开为 2B 行序列（偶数行 chosen，奇数行 rejected），右 padding。"""
    rows: list[tuple[list[int], int]] = []
    for ex in batch:
        for key in ("chosen", "rejected"):
            part = ex[key]
            rows.append((part["prompt_ids"] + part["resp_ids"], len(part["prompt_ids"])))
    maxlen = max(len(ids) for ids, _ in rows)
    bsz = len(rows)
    input_ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, maxlen), dtype=torch.long)
    resp_mask = torch.zeros((bsz, maxlen), dtype=torch.bool)
    for i, (ids, plen) in enumerate(rows):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        resp_mask[i, plen:length] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "resp_mask": resp_mask}


class DPOTrainer:
    """手写 DPO 循环：policy = 当前模型 + LoRA；reference = disable_adapter 的基座。

    S3 契约：model 必须是 peft PeftModel（reference 依赖 disable_adapter 上下文），
    非 PeftModel 在 __init__ 直接 raise RuntimeError——静默退化 ref=policy 会让
    参考项恒为 0、DPO 失效为类 CPO，属于必须拒绝的错误配置
    （7B 全参 DPO 必然 OOM，validate_config 已要求 dpo_enabled ⇒ lora.enabled）。
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        output_dir: Path,
        logger=None,
        *,
        length_norm: bool = False,
    ):
        """
        参数：
            model: peft PeftModel（bf16 基座 + 活跃 LoRA adapter；只训练 LoRA 参数）
            tokenizer: HF tokenizer（须有 pad_token_id）
            config: TrainingConfig（使用 dpo_beta / dpo_epochs / dpo_batch_size /
                dpo_grad_accum / dpo_lr / max_steps 字段）
            output_dir: checkpoint 输出根目录（dpo_epoch{n}/ 子目录，仅 adapter）
            length_norm: 序列 logprob 是否按 response token 数归一（默认 False）
        """
        if not isinstance(model, PeftModel):
            raise RuntimeError(
                "DPOTrainer 需要 peft PeftModel（reference 经 disable_adapter() 取"
                "基座分布），收到 "
                f"{type(model).__name__}。请传入 adapter_manager.model；"
                "7B 全参 DPO 必然 OOM（dpo_enabled 要求 lora.enabled），"
                "禁止静默退化 ref=policy"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("debate_erc.training.dpo")
        self.device = next(model.parameters()).device
        self.length_norm = length_norm

    def train(self, pairs: list[PreferencePair]) -> dict:
        """执行 DPO 训练，返回摘要 dict。

        返回：{"steps": int, "loss": float, "margins": float, "implicit_acc": float}
            loss / margins / implicit_acc 均为 micro-batch 均值；
            margins = β*((π_c-ref_c)-(π_r-ref_r)) 的平均（reward margin），
            implicit_acc = chosen 排名正确率（margins > 0 的比例）。
        """
        summary: dict = {"steps": 0, "loss": 0.0, "margins": 0.0, "implicit_acc": 0.0}
        if not pairs:
            self.logger.warning("DPOTrainer.train() 收到空偏好对列表，直接返回")
            return summary
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("模型无可训练参数（DPO 需要 LoRA adapter 处于可训练状态）")
        optimizer = torch.optim.AdamW(params, lr=self.config.dpo_lr)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        dataset = _DPODataset(pairs, self.tokenizer, logger=self.logger)
        loader = DataLoader(
            dataset, batch_size=self.config.dpo_batch_size, shuffle=True, drop_last=False,
            collate_fn=partial(_collate, pad_id=pad_id),
        )
        beta = self.config.dpo_beta

        global_step = 0
        loss_sum, margin_sum, acc_sum, cnt = 0.0, 0.0, 0.0, 0
        stop = False
        for epoch in range(self.config.dpo_epochs):
            if stop:
                break
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            truncated = False
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                resp_mask = batch["resp_mask"]
                # ---- policy（含活跃 LoRA，带梯度）----
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                pi = self._response_logprob(logits, input_ids, resp_mask)
                # ---- reference（disable_adapter 的基座；eval() 包裹防 dropout 污染）----
                with torch.no_grad():
                    was_training = self.model.training
                    self.model.eval()
                    try:
                        with self.model.disable_adapter():
                            ref_logits = self.model(
                                input_ids=input_ids, attention_mask=attention_mask
                            ).logits
                        ref = self._response_logprob(ref_logits, input_ids, resp_mask)
                    finally:
                        if was_training:
                            self.model.train()
                # ---- DPO 损失（交错布局：偶 chosen / 奇 rejected）----
                pi_c, pi_r = pi[0::2], pi[1::2]
                ref_c, ref_r = ref[0::2], ref[1::2]
                margins = beta * ((pi_c - ref_c) - (pi_r - ref_r))
                loss = -F.logsigmoid(margins).mean()
                (loss / self.config.dpo_grad_accum).backward()
                loss_sum += float(loss.item())
                margin_sum += float(margins.mean().item())
                acc_sum += float((margins > 0).float().mean().item())
                cnt += 1
                micro += 1
                if micro % self.config.dpo_grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    self._log_memory(global_step)
                    if 0 < self.config.max_steps <= global_step:
                        stop = True
                        truncated = True  # epoch 中途截断（R2-m6：跳过 epoch 落盘）
                        break
            if not stop and micro and micro % self.config.dpo_grad_accum != 0:
                k = micro % self.config.dpo_grad_accum
                # 梯度补偿：残余 k 个 micro-batch 的梯度均值口径与完整周期一致
                #（与 SFTTrainer 同构，S5 契约）
                scale = self.config.dpo_grad_accum / k
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(scale)
                optimizer.step()  # epoch 末冲刷残余梯度
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                self._log_memory(global_step)
                if 0 < self.config.max_steps <= global_step:
                    stop = True
            if truncated:
                # max_steps 截断：epoch 未完成，跳过收尾（与 SFTTrainer 同口径，
                # R2-m6：避免未跑完的 epoch 以 dpo_epoch{n} 落盘被误读为完整 epoch）
                self.logger.info(
                    "[dpo] epoch %d 被 max_steps 截断于 step %d，跳过 epoch 落盘",
                    epoch + 1, global_step,
                )
                break
            # epoch 落盘：PeftModel.save_pretrained 只写 adapter 权重
            #（7B 主干不落盘）；selected_adapters 限定当前活跃 adapter
            self.model.save_pretrained(
                self.output_dir / f"dpo_epoch{epoch + 1}",
                selected_adapters=self.model.active_adapters,
            )
            self.logger.info(
                "[dpo] epoch %d done: steps=%d avg_loss=%.4f avg_margin=%.4f implicit_acc=%.3f",
                epoch + 1, global_step,
                loss_sum / max(cnt, 1), margin_sum / max(cnt, 1), acc_sum / max(cnt, 1),
            )
        if cnt:
            summary.update({
                "steps": global_step,
                "loss": loss_sum / cnt,
                "margins": margin_sum / cnt,
                "implicit_acc": acc_sum / cnt,
            })
        return summary

    def _response_logprob(
        self, logits: torch.Tensor, input_ids: torch.Tensor, resp_mask: torch.Tensor
    ) -> torch.Tensor:
        """response token 的序列 logprob（shift 对齐），可选按 token 数归一。"""
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:]
        mask = resp_mask[:, 1:].to(logp.dtype)
        tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [2B, T-1]
        seq_logp = (tok_logp * mask).sum(dim=-1)
        if self.length_norm:
            seq_logp = seq_logp / mask.sum(dim=-1).clamp(min=1.0)
        return seq_logp

    def _log_memory(self, step: int) -> None:
        """显存护栏：每 50 个优化器步记录峰值显存。"""
        if step % 50 == 0 and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.logger.info("[mem] step=%d max_memory_allocated=%.2f GiB", step, peak)
