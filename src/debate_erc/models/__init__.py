"""模型层：主干加载、LoRA 工厂、adapter 管理。"""

from .adapter_manager import AdapterManager
from .backbone import LLMGenerator, load_backbone
from .lora_factory import TARGET_PRESETS, build_lora_config, count_trainable

__all__ = [
    "load_backbone", "LLMGenerator", "build_lora_config",
    "count_trainable", "AdapterManager", "TARGET_PRESETS",
]
