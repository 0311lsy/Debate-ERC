"""AdapterManager 加载护栏测试（Layer 0 PEFT 同名静默跳过事故的回归测试）。

事故背景：离线 A/B 对照先 add("main") 再 model.load_adapter(dir, "main")，
PEFT 对同名 adapter 静默跳过、不替换权重、不报错，导致三个 checkpoint
读到同一权重、全部对照结论作废。
"""

from __future__ import annotations

import pytest

from debate_erc.config import LoRAConfig
from debate_erc.models.adapter_manager import AdapterManager


class FakePeftModel:
    """非 PeftModel 时 add/load 才会真正创建；这些用例只验证注册层护栏，
    故让 isinstance 检查命中非 PeftModel 分支前先拦截——通过 monkey patch
    管理器构造函数不现实，改为直接操作 adapters 注册表模拟状态。"""

    def set_adapter(self, name):
        pass


class TestAdapterManagerGuardrails:
    def _fresh(self):
        mgr = AdapterManager(model=object())
        return mgr

    def test_duplicate_add_rejected(self, monkeypatch):
        """同名 add 必须报错（而非静默复用旧权重）。"""
        mgr = self._fresh()
        # 模拟首个 adapter 已创建成功（绕过真实 PEFT 初始化）
        monkeypatch.setattr(
            "debate_erc.models.adapter_manager.get_peft_model",
            lambda model, cfg, adapter_name: FakePeftModel(),
        )
        mgr.add("main", LoRAConfig())
        with pytest.raises(ValueError, match="已存在"):
            mgr.add("main", LoRAConfig())

    def test_load_existing_name_rejected(self):
        """对已注册名 load 必须报错（PEFT 静默跳过的直接防护）。"""
        mgr = self._fresh()
        mgr.adapters["main"] = LoRAConfig()
        with pytest.raises(ValueError, match="已存在"):
            mgr.load("main", "/nonexistent/should_not_reach")

    def test_load_missing_dir_rejected(self, tmp_path):
        """无 adapter_config.json 的目录显式报错。"""
        mgr = self._fresh()
        with pytest.raises(FileNotFoundError):
            mgr.load("ext", tmp_path)
