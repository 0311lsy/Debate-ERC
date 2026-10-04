"""SFT 训练器（V3 6.2 / 设计报告 5.5）：纯 PyTorch 训练循环（不用 trl）。

为什么手写循环：复现教训（V3 八）trl Trainer 的 collator 闭包在大数据集下
持有样本引用导致 OOM；本实现按 batch 现构建 tensor、用完即释放，规避闭包问题。

核心约定：
- 每样本 prompt = build_classify_prompt(context, schema)，answer = " {gold_label}\\n"
- input_ids = prompt_ids + answer_ids；labels 仅 answer 部分非 -100（prompt/pad 掩 -100）
- 记录 answer 中标签词首 token 的绝对位置与类别索引（供 focal / 类别加权）
- loss：全部 answer token 普通 CE；label 词首 token 位置替换为 focal / 类别加权版本
  （V3 7.3：focal_gamma 与 class_weights 可组合；focal 路径按 S5 展平契约传入
  已展平 [N, V] float logits，避免 reshape+float 双份复制）
- bf16 主干直接训练 LoRA 参数（不额外 autocast）；AdamW + 梯度累积
  （epoch 末残余梯度 flush 前做 grad_accum/k 梯度补偿）
- 辅助任务：本训练器只接受 DialogueSample 列表，多任务由 multitask_scheduler
  以外层交替方式调用（推荐：外层交替，简单可靠）
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from ..agents.prompt_builder import build_classify_prompt
from ..config import TrainingConfig
from ..data.schemas import DialogueSample
from ..fewshot.focal_loss import FocalLabelWeightedCE
from ..utils.label_schema import LabelSchema
from ..utils.logging import get_logger

_MAX_LEN = 2048          # 与 backbone.LLMGenerator 的 max_length 口径一致
_MAX_ANSWER_TOKENS = 32  # " {label}\\n" 远小于此


def left_truncate_with_bos(
    prompt_ids: list[int], resp_len: int, bos_id: int | None, max_len: int
) -> tuple[list[int], int]:
    """超长样本左截断 prompt（保尾，保留含目标话语的末尾）并重新前置 BOS。

    左截断会把 tokenizer 加上的句首 BOS 一并截掉；因果 LM 的条件分布以 BOS
    开头为准，因此截断后需重新前置（S5/S7 同构契约，SFT 与 DPO 共用）。
    预留 BOS 位保证前置后总长仍 ≤ max_len（从头部多让 1 个位置，不伤保尾）。

    返回 (截断后的 prompt_ids, 截断前总长)；未超长时原样返回。
    调用方据「截断前总长 > max_len」打 warning 日志。
    """
    total = len(prompt_ids) + resp_len
    if total <= max_len:
        return prompt_ids, total
    keep = max(1, max_len - resp_len - (1 if bos_id is not None else 0))
    truncated = prompt_ids[-keep:]
    if bos_id is not None and (not truncated or truncated[0] != bos_id):
        truncated = [bos_id] + truncated
    return truncated, total


def _label_token_offset(tokenizer, answer_ids: list[int]) -> int | None:
    """answer 序列中标签词首 token 的相对下标（跳过纯空白 token；找不到返回 None）。"""
    for i, tid in enumerate(answer_ids):
        text = tokenizer.decode([tid])
        if text and text.strip():
            return i
    return None


class _SFTDataset(Dataset):
    """DialogueSample 列表 → 预编码样本（input_ids + 边界 + label 位置/类别）。

    prompt_builder 可注入（辅助任务用 intensity/speaker 提示，与 classify.j2 解耦）。
    """

    def __init__(
        self,
        samples: list[DialogueSample],
        tokenizer,
        schema: LabelSchema,
        prompt_builder=None,
        logger=None,
    ):
        build_prompt = prompt_builder or build_classify_prompt
        bos_id = getattr(tokenizer, "bos_token_id", None)
        self.examples: list[dict] = []
        for s in samples:
            prompt = build_prompt(s.context, schema)
            answer = f" {s.gold_label}\n"
            # prompt 不做 tokenizer 右截断（R2-m4）：truncation=True 会先砍掉
            # prompt 尾部（含目标话语），架空 left_truncate_with_bos 的保尾语义；
            # 超长保护全权交给下方 left_truncate_with_bos
            prompt_ids = list(
                tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids
            )
            answer_ids = list(
                tokenizer(answer, add_special_tokens=False, truncation=True,
                          max_length=_MAX_ANSWER_TOKENS).input_ids
            )
            # 总长保护：超长时左截断 prompt（保尾）并重新前置 BOS
            prompt_ids, orig_len = left_truncate_with_bos(
                prompt_ids, len(answer_ids), bos_id, _MAX_LEN
            )
            if orig_len > _MAX_LEN and logger is not None:
                logger.warning(
                    "[sft] 样本超长（%d > %d tokens）：prompt 已左截断并重新前置 BOS"
                    "（dialogue_id=%s）",
                    orig_len, _MAX_LEN, s.context.dialogue_id,
                )
            off = _label_token_offset(tokenizer, answer_ids)
            self.examples.append({
                "input_ids": prompt_ids + answer_ids,
                "prompt_len": len(prompt_ids),
                "label_pos": len(prompt_ids) + off if off is not None else -1,
                "label_class": schema.index(s.gold_label),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def _collate(batch: list[dict], pad_id: int) -> dict:
    """右 padding；labels 的 prompt/pad 部分为 -100。"""
    maxlen = max(len(ex["input_ids"]) for ex in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, maxlen), dtype=torch.long)
    labels = torch.full((bsz, maxlen), -100, dtype=torch.long)
    label_pos = torch.full((bsz,), -1, dtype=torch.long)
    label_class = torch.zeros((bsz,), dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        plen = ex["prompt_len"]
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        labels[i, plen:length] = torch.tensor(ids[plen:], dtype=torch.long)
        label_pos[i] = ex["label_pos"]
        label_class[i] = ex["label_class"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_pos": label_pos,
        "label_class": label_class,
    }


class SFTTrainer:
    """监督微调训练器（纯 PyTorch 循环，规避 trl 闭包 OOM 教训）。"""

    def __init__(
        self,
        model,
        tokenizer,
        schema: LabelSchema,
        config: TrainingConfig,
        output_dir: Path,
        adapter_manager=None,
        class_weights=None,
        focal_gamma: float | None = None,
        prompt_builder=None,
        logger=None,
    ):
        """
        参数：
            model: 因果 LM（bf16 加载，含可训练 LoRA 参数；直接训练，无 autocast）
            tokenizer: HF tokenizer（须有 pad_token_id，backbone 加载护栏已保证）
            schema: 标签体系
            config: TrainingConfig
            output_dir: checkpoint 输出根目录
            adapter_manager: 可选 AdapterManager（保存活跃 adapter 用）
            class_weights: 类别权重，dict[label, float]（class_balanced_weights 输出）
                或按 schema.labels 顺序的 list / Tensor；None → 不加权
            focal_gamma: focal 调制指数（V3 7.3，默认 2.0 由调用方显式传入）；None → 不启用
            prompt_builder: 可选 prompt 构建（context, schema）→ str；
                None → build_classify_prompt（辅助任务注入各自的模板）
            logger: 可选外部 logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.schema = schema
        self.config = config
        self.output_dir = Path(output_dir)
        self.adapter_manager = adapter_manager
        self.prompt_builder = prompt_builder
        self.logger = logger or get_logger("debate_erc.training.sft")
        self.device = next(model.parameters()).device
        self.criterion = self._build_criterion(class_weights, focal_gamma)

    def _build_criterion(self, class_weights, focal_gamma) -> FocalLabelWeightedCE | None:
        """两者都为 None → 纯 CE；否则 FocalLabelWeightedCE（γ=0 退化为类别加权 CE）。"""
        if focal_gamma is None and class_weights is None:
            return None
        cw: torch.Tensor | None = None
        if class_weights is not None:
            if isinstance(class_weights, dict):
                cw = torch.tensor(
                    [float(class_weights.get(lb, 1.0)) for lb in self.schema.labels],
                    dtype=torch.float32,
                )
            else:
                cw = torch.as_tensor(class_weights, dtype=torch.float32)
        gamma = float(focal_gamma) if focal_gamma is not None else 0.0
        return FocalLabelWeightedCE(gamma=gamma, class_weights=cw)

    # ------------------------------------------------------------------
    # Dataset 构建（公开接口，S5 契约）
    # ------------------------------------------------------------------

    def build_dataset(self, samples: list[DialogueSample]) -> _SFTDataset:
        """把 DialogueSample 列表预编码为训练 dataset（公开接口，S5 契约）。

        multitask_scheduler 在首个 chunk 前调用一次，后续 chunk 经
        train(..., dataset=...) 复用同一 dataset，避免每个 chunk 重复
        tokenize 全量样本（tokenizer 编码在长训练中重复 N 次是纯浪费）。
        """
        return _SFTDataset(
            samples, self.tokenizer, self.schema,
            prompt_builder=self.prompt_builder, logger=self.logger,
        )

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self, samples: list[DialogueSample], eval_fn=None,
              steps_budget: int | None = None, optimizer=None, dataset=None) -> dict:
        """执行训练，返回摘要 dict。

        参数：
            samples: DialogueSample 列表。辅助任务（intensity 等）由 multitask_scheduler
                以外层交替方式实现——本训练器不接受 (sample, aux_target) 元组，
                传入 tuple 会抛 TypeError（保持训练循环简单可靠）。
            eval_fn: 可选回调 eval_fn(model)，每 epoch 结束调用；且
                config.eval_every_steps > 0 时按 global_step 触发（S5 契约）。
                返回值（如 dev W-F1）记入 eval_history 并落日志。
                调用前 model.eval()、后恢复 train()（防 dropout 污染评估）。
            steps_budget: 本次调用最多执行的优化器步数（multitask_scheduler 分块
                交替训练用）；None → 不限制（受 config.max_steps 约束）。
                触发截断的 epoch 视为未完成：跳过收尾（不 save checkpoint_epoch{n}、
                不计入 epochs_done）。
            optimizer: 外部传入的优化器（跨 chunk 复用 AdamW 状态）；None → 新建。
            dataset: 预构建的训练 dataset（build_dataset 输出，S5 契约）；None →
                现场构建。multitask_scheduler 分块调用时传入，避免每 chunk 重复
                tokenize 全量样本。

        返回：{"steps": int, "final_loss": float, "epochs_done": int, "eval_history": list}
            final_loss 为全部 micro-batch loss 的均值（跨 epoch 累计）；
            epoch 日志中的 avg_loss 为当轮均值。

        lr warmup（S5）：get_linear_schedule_with_warmup，
        num_training_steps = steps_budget or 估算总步数（估算 = 每 epoch 优化器步数
        × epochs，再受 max_steps 封顶），warmup = warmup_ratio × num_training_steps。
        注意 chunk 模式（multitask_scheduler 分块）下按 chunk 近似：每个 chunk 以
        steps_budget 为总步数重建调度器，warmup 在每个 chunk 内重新升温。
        """
        if samples and not isinstance(samples[0], DialogueSample):
            if isinstance(samples[0], tuple):
                raise TypeError(
                    "SFTTrainer.train 只接受 DialogueSample 列表；辅助任务请由 "
                    "multitask_scheduler 以外层交替方式调用本训练器"
                )
            raise TypeError(
                f"samples 元素必须是 DialogueSample，收到 {type(samples[0]).__name__}"
            )
        summary: dict = {"steps": 0, "final_loss": float("nan"),
                         "epochs_done": 0, "eval_history": []}
        if not samples:
            self.logger.warning("train() 收到空样本列表，直接返回")
            return summary
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("模型无可训练参数（LoRA 未启用？检查 lora.enabled / adapter 注册）")
        if optimizer is None:
            optimizer = torch.optim.AdamW(params, lr=self.config.lr)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if dataset is None:
            dataset = self.build_dataset(samples)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False,
            collate_fn=partial(_collate, pad_id=pad_id),
        )

        # ---- lr warmup 调度（S5）----
        n_batches = math.ceil(len(dataset) / self.config.batch_size)
        steps_per_epoch = math.ceil(n_batches / self.config.grad_accum)
        est_total = steps_per_epoch * self.config.epochs
        if 0 < self.config.max_steps < est_total:
            est_total = self.config.max_steps
        num_training_steps = steps_budget if steps_budget else est_total
        warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )

        def budget_reached(step: int) -> bool:
            return (0 < self.config.max_steps <= step) or (
                steps_budget is not None and step >= steps_budget
            )

        global_step = 0
        loss_sum, loss_cnt = 0.0, 0  # 全局累计（final_loss 口径）
        stop = False
        for epoch in range(self.config.epochs):
            if stop:
                break
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum, epoch_loss_cnt = 0.0, 0  # 当轮均值（epoch 日志口径）
            micro = 0
            truncated = False
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = self._compute_loss(logits, batch)
                (loss / self.config.grad_accum).backward()
                loss_sum += float(loss.item())
                loss_cnt += 1
                epoch_loss_sum += float(loss.item())
                epoch_loss_cnt += 1
                micro += 1
                if micro % self.config.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    self._log_memory(global_step)
                    self._maybe_eval(eval_fn, global_step, summary)
                    if budget_reached(global_step):
                        stop = True  # max_steps / steps_budget 截断（smoke 或分块交替训练）
                        truncated = True
                        break
            if not stop and micro and micro % self.config.grad_accum != 0:
                k = micro % self.config.grad_accum
                # 梯度补偿：残余 k 个 micro-batch 的 (loss/grad_accum).backward()
                # 梯度和偏小 grad_accum/k 倍，flush 前乘回，保证与完整周期口径一致
                scale = self.config.grad_accum / k
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(scale)
                optimizer.step()  # epoch 末冲刷不足一个累积周期的梯度
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                self._log_memory(global_step)
                self._maybe_eval(eval_fn, global_step, summary)
                if budget_reached(global_step):
                    stop = True
            if truncated:
                # steps_budget 截断：epoch 未完成，跳过收尾
                #（不 save checkpoint_epoch{n}、不计入 epochs_done）
                self.logger.info(
                    "[sft] epoch %d 被 steps_budget/max_steps 截断于 step %d，跳过 epoch 收尾",
                    epoch + 1, global_step,
                )
                break
            summary["epochs_done"] = epoch + 1
            self.logger.info(
                "[sft] epoch %d done: steps=%d avg_loss=%.4f",
                epoch + 1, global_step, epoch_loss_sum / max(epoch_loss_cnt, 1),
            )
            if self.config.save_every_epoch:
                self._save_checkpoint(self.output_dir / f"checkpoint_epoch{epoch + 1}")
            if eval_fn is not None:
                result = self._run_eval(eval_fn)
                if result is not None:
                    self.logger.info("[sft] epoch %d eval: %s", epoch + 1, result)
                    summary["eval_history"].append(result)

        summary["steps"] = global_step
        summary["final_loss"] = loss_sum / max(loss_cnt, 1)
        return summary

    def _maybe_eval(self, eval_fn, global_step: int, summary: dict) -> None:
        """eval_every_steps > 0 且传入 eval_fn 时按 global_step 触发评估（S5 契约）。"""
        if eval_fn is None or self.config.eval_every_steps <= 0:
            return
        if global_step % self.config.eval_every_steps != 0:
            return
        result = self._run_eval(eval_fn)
        if result is not None:
            self.logger.info("[sft] step %d eval: %s", global_step, result)
            summary["eval_history"].append(result)

    def _run_eval(self, eval_fn):
        """eval_fn 调用前 model.eval()、后恢复 train()（防 dropout 污染评估）。"""
        was_training = self.model.training
        self.model.eval()
        try:
            return eval_fn(self.model)
        finally:
            if was_training:
                self.model.train()

    def _compute_loss(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        """全部 answer token 普通 CE；label 词首 token 位置替换为 focal / 类别加权 CE。

        focal 路径按 S5 展平契约：从唯一一份展平 float logits（flat，普通 CE
        已构建）中取 label 行传入 criterion，criterion 内不再 reshape+float。
        """
        labels = batch["labels"]
        bsz, _tlen = labels.shape
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat = shift_logits.reshape(-1, shift_logits.size(-1)).float()
        ce = F.cross_entropy(flat, shift_labels.reshape(-1), reduction="none",
                             ignore_index=-100)
        ce = ce.reshape(bsz, -1)  # [B, T-1]，忽略位置为 0
        valid = shift_labels != -100
        n_valid = int(valid.sum())
        if n_valid == 0:
            return logits.sum() * 0.0
        if self.criterion is None:
            return ce.sum() / n_valid

        # label 词首 token 在 shift 后坐标系的位置 = label_pos - 1
        pos = batch["label_pos"] - 1
        seq = shift_labels.size(1)
        rows = torch.arange(bsz, device=labels.device)
        safe = pos.clamp(0, seq - 1)
        tok = shift_labels[rows, safe]
        has_label = (pos >= 0) & (pos < seq) & (tok != -100)
        n_label = int(has_label.sum())
        if n_label == 0:
            return ce.sum() / n_valid
        # 从 flat（[B*(T-1), V]，免复制的 view）中取 label 行 → criterion 收到
        # 已展平 [n_label, V] float logits + 展平 targets（S5 契约）
        flat_seq = flat.view(bsz, seq, -1)
        sel_logits = flat_seq[rows[has_label], safe[has_label]]  # [n_label, V] float
        sel_targets = tok[has_label]                              # [n_label]
        sel_classes = batch["label_class"][has_label]              # [n_label]
        # criterion 在 label 位置上返回均值 → ×n_label 还原成 sum，与整体 per-token 均值合并
        label_loss_sum = self.criterion(sel_logits, sel_targets, sel_classes) * n_label
        plain_label_sum = ce[rows[has_label], safe[has_label]].sum()
        return (ce.sum() - plain_label_sum + label_loss_sum) / n_valid

    # ------------------------------------------------------------------
    # 护栏与保存
    # ------------------------------------------------------------------

    def _log_memory(self, step: int) -> None:
        """显存护栏：每 50 个优化器步记录峰值显存（V3 八 复现教训）。"""
        if step % 50 == 0 and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.logger.info("[mem] step=%d max_memory_allocated=%.2f GiB", step, peak)

    def _save_checkpoint(self, path: Path) -> None:
        """保存 adapter：优先经 adapter_manager（保存活跃 adapter），否则 model.save_pretrained。"""
        path = Path(path)
        if self.adapter_manager is not None and self.adapter_manager.active is not None:
            self.adapter_manager.save(self.adapter_manager.active, path)
        else:
            self.model.save_pretrained(path)
        self.logger.info("[sft] checkpoint 已保存: %s", path)
