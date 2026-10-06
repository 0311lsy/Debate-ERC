#!/usr/bin/env python
"""门控叠加 InstructERC proponent：提取 LoRA adapter → 打分 p_a → 冻结门控。

从 InstructERC epoch8 的 26GB DeepSpeed checkpoint 中提取 LoRA 权重，
保存为标准 PEFT adapter 格式，避免每次加载完整模型。
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

SRC = Path("/home/lsy20252770/Agent_Reason/InstructERC/results/meld_lora_10epoch/epoch_8/mp_rank_00_model_states.pt")
DST = Path("/home/lsy20252770/Agent_Reason/Debate-ERC/outputs/instructerc_adapter_ep8")

def main():
    DST.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint from {SRC} ...")
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    sd = ckpt["module"]

    # 提取 LoRA 权重
    # PEFT 加载时会调用 _insert_adapter_name_into_state_dict，
    # 在 "lora_" 之后的 suffix 中插入 adapter_name（default）。
    # 原始 checkpoint 键名已含 ".default."，若原样保存会导致双写
    # （lora_A.default.default.weight），全部被 miss。
    # 因此提取时必须先 strip ".default" 段，让 PEFT 自行插入。
    lora_sd = {}
    for k, v in sd.items():
        if "lora" in k.lower():
            new_key = k.replace(".default.", ".")
            lora_sd[new_key] = v.to(torch.float16)

    print(f"Extracted {len(lora_sd)} LoRA tensors")

    # adapter_config.json
    config = {
        "base_model_name_or_path": "/home/lsy20252770/InstructERC/LLM_bases/LLaMA2",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "revision": None,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    (DST / "adapter_config.json").write_text(json.dumps(config, indent=2))

    # adapter_model.safetensors
    save_file(lora_sd, str(DST / "adapter_model.safetensors"))
    print(f"Adapter saved to {DST}")
    total_bytes = sum(v.numel() * v.element_size() for v in lora_sd.values())
    print(f"Adapter size: {total_bytes / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()
