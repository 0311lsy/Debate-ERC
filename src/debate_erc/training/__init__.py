"""训练层：SFT / DPO 训练器与偏好对构建（V3 6.2 / 设计报告 5.5）。"""

from __future__ import annotations

from .dpo_trainer import DPOTrainer
from .preference_pair import GoldLabelPairBuilder, PairSource, build_pairs
from .sft_trainer import SFTTrainer

__all__ = [
    "SFTTrainer",
    "DPOTrainer",
    "PairSource",
    "GoldLabelPairBuilder",
    "build_pairs",
]
