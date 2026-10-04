"""Focal Loss（V3 7.3）：标签 token 位置的 focal 调制 + 类别加权交叉熵。

- focal_ce：纯函数，展平 token 位置集合上的 focal CE（S5 契约：接受已展平
  [N, V] float logits，调用方负责展平/转 float，避免接口内二次复制整份激活）；
- FocalLabelWeightedCE：面向 SFT 的封装，label 词首 token 位置 CE，
  focal 调制 (1-p_t)^γ，class_weights 按该样本类别加权。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_IGNORE_INDEX = -100


def focal_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight_per_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """纯函数 focal CE（S5 展平契约：调用方传已展平输入）。

    参数：
        logits: [N, V] 已展平的 float logits（调用方负责 reshape/float，
            避免 [B, T, V] 整份激活在此二次复制）
        targets: [N] 展平目标 token id，-100 位置被忽略
        gamma: focal 调制指数（0 → 退化为普通加权 CE）
        weight_per_token: [N] 逐 token 权重，None → 全 1

    返回：
        标量损失 = valid 位置上 mean( (1-p_t)^γ * CE * w )，
        p_t = softmax(logits)[target]。
    """
    if logits.dim() != 2:
        raise ValueError(
            f"focal_ce 期望已展平的 [N, V] logits，收到 shape {tuple(logits.shape)}"
        )
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"logits {tuple(logits.shape)} 与 targets {tuple(targets.shape)} 位置维度不一致"
        )
    valid = targets != _IGNORE_INDEX
    n_valid = int(valid.sum())
    if n_valid == 0:
        return logits.sum() * 0.0  # 无有效位置：返回保持计算图的 0
    ce = F.cross_entropy(
        logits, targets, reduction="none", ignore_index=_IGNORE_INDEX
    )  # 忽略位置返回 0
    p_t = torch.exp(-ce)  # 目标类概率
    focal = (1.0 - p_t).clamp(min=0.0) ** gamma
    if weight_per_token is None:
        w = torch.ones_like(ce)
    else:
        w = weight_per_token.to(device=ce.device, dtype=ce.dtype)
    loss = ce * focal * w
    return loss[valid].sum() / n_valid


class FocalLabelWeightedCE(torch.nn.Module):
    """标签 token 加权 focal CE（V3 7.3；S5 展平契约）。

    forward 约定（展平输入，调用方负责展平/筛选）：
        logits: [N, V] 已展平的 float logits（N = 有效 label 位置数）
        targets: [N]，展平目标 token id（-100 位置由调用方过滤/保留皆可）
        label_class: [N] 每个位置所属样本的类别索引（对应 class_weights 下标）

    class_weights: [num_classes]，顺序须与 LabelSchema.labels 一致
    （可由 fewshot.class_balance.class_balanced_weights 的 dict 按 schema 顺序展开）。
    """

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_class: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """label token 位置的 focal + 类别加权 CE（标量，valid 位置均值）。"""
        if self.class_weights is None or label_class is None:
            weight = None
        else:
            lc = label_class.to(logits.device).long()
            weight = self.class_weights.to(logits.device)[lc]  # [N] 逐位置权重
        return focal_ce(logits, targets, self.gamma, weight)
