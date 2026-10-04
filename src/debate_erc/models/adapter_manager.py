"""AdapterManager：任务特定 LoRA adapter 管理（V3 5.5 / 6.8.3）。

- 主任务 adapter "main"（r=64）与辅助任务 adapter（如 "intensity", r=32）相互独立
- activate(name) 切换当前活跃 adapter，训练互不污染
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel, get_peft_model

from ..config import LoRAConfig
from .lora_factory import build_lora_config, count_trainable


class AdapterManager:
    """多 adapter 生命周期管理。"""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters: dict[str, LoRAConfig] = {}
        self.active: str | None = None

    def add(self, name: str, lora_config: LoRAConfig) -> "AdapterManager":
        """注册并添加一个 adapter（首个自动成为活跃 adapter）。

        Layer 0 护栏：同名重复注册显式报错。PEFT 的 load_adapter 对已存在的
        同名 adapter 会**静默跳过、不替换权重、不报错**（曾导致离线 A/B 对照
        全部读到同一权重、结论作废），因此任何创建路径都不允许同名复用；
        需要覆盖时先 delete_adapter 或改用唯一名称。
        """
        if name in self.adapters:
            raise ValueError(
                f"adapter {name!r} 已存在；禁止同名重新加载（PEFT 会静默跳过"
                f"导致读到旧权重）。请使用唯一名称或先删除旧 adapter。"
            )
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, build_lora_config(lora_config), adapter_name=name)
        else:
            self.model.add_adapter(name, build_lora_config(lora_config))
        self.adapters[name] = lora_config
        if self.active is None:
            self.active = name
            self.model.set_adapter(name)
        return self

    def load(self, name: str, path: str | Path) -> "AdapterManager":
        """从磁盘安全加载一个已训练 adapter（评估/离线对照专用）。

        防护 PEFT 同名静默跳过：name 必须不存在；加载后校验该 adapter 名下
        确实挂载了非空 LoRA 参数（防空目录/损坏权重被静默接受）。
        """
        path = Path(path)
        if name in self.adapters:
            raise ValueError(
                f"adapter {name!r} 已存在；load_adapter 同名时 PEFT 静默跳过、"
                f"不会替换权重。请改用唯一 adapter 名。"
            )
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"{path} 下无 adapter_config.json，不是有效 adapter 目录")
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
        else:
            self.model.load_adapter(str(path), adapter_name=name)
        self.adapters[name] = None  # 外部加载的 adapter 无构造配置
        # 校验：该名称名下确有非空参数（PEFT 参数名以 adapter 名为后缀段）
        n_params = 0
        for pname, param in self.model.named_parameters():
            if ".lora_" in pname and pname.split(".")[-2] == name:
                if param.numel() > 0:
                    n_params += 1
        if n_params == 0:
            raise RuntimeError(f"从 {path} 加载 {name!r} 后未发现任何 LoRA 参数，加载可能静默失败")
        if self.active is None:
            self.active = name
            self.model.set_adapter(name)
        return self

    def activate(self, name: str) -> None:
        """切换活跃 adapter；切换前后可训练参数集合互不污染（验收 2.4-②）。

        set_adapter 只切换前向路径；这里额外同步 requires_grad：
        仅活跃 adapter 的 LoRA 参数可训练，其余 adapter 全部冻结，
        保证任务特定 adapter 训练互不污染（V3 5.5）。
        """
        if name not in self.adapters:
            raise KeyError(f"adapter 未注册: {name}，已有 {list(self.adapters)}")
        self.model.set_adapter(name)
        self.active = name
        self._sync_trainability()

    def _sync_trainability(self) -> None:
        """PEFT 参数名形如 ...lora_A.{adapter_name}.weight，按 adapter 名同步 requires_grad。"""
        for pname, param in self.model.named_parameters():
            if ".lora_" in pname:
                adapter_name = pname.split(".")[-2]
                param.requires_grad = adapter_name == self.active

    def trainable_params(self) -> tuple[int, int, float]:
        return count_trainable(self.model)

    def save(self, name: str, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        current = self.active
        self.activate(name)
        self.model.save_pretrained(path, selected_adapters=[name])
        if current and current != name:
            self.activate(current)

    def merge_and_unload(self):
        return self.model.merge_and_unload()
