"""LoRA 配置工厂（设计报告 4.2）：build_lora_config(r, alpha, dropout, targets)。"""

from __future__ import annotations

from peft import LoraConfig

from ..config import LoRAConfig

# 目标模块预设（消融开关 lora.target_modules = qv | qkvo）
TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "qv": ("q_proj", "v_proj"),
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def build_lora_config(lora: LoRAConfig) -> LoraConfig:
    """把项目内 LoRAConfig 转为 peft LoraConfig（统一单一事实来源）。"""
    if isinstance(lora.target_modules, str):
        preset = lora.target_modules
        if preset not in TARGET_PRESETS:
            raise ValueError(
                f"未知 target_modules 预设 {preset!r}；合法预设: {sorted(TARGET_PRESETS)}。"
                "如需自定义目标模块，请直接传模块名列表/元组，如 ('q_proj', 'v_proj')"
            )
        targets = TARGET_PRESETS[preset]
    else:
        targets = tuple(lora.target_modules)
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(targets),
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_trainable(model) -> tuple[int, int, float]:
    """返回 (可训练参数量, 总参数量, 占比)。验收 0.3-②：≈79.9M / 1.17%。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / max(total, 1)
