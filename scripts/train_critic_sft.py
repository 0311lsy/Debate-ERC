#!/usr/bin/env python
"""异构 critic 训练：DeepSeek-R1-Distill-Qwen-1.5B + LoRA 在 MELD 上纯 SFT。

公平性控制（审稿口径）：与 m0_sft（LLaMA2-7B proponent）完全相同的
- 训练数据（MELD train 9989，无合成/类平衡/focal/辅助）
- 提示模板（classify.j2，SFT 目标 " {gold}\\n"）
- 超参（3 epoch / 有效 batch 32 / lr 2e-4 / warmup 0.03）
- LoRA 规格（r=64, alpha=128, dropout=0.05, q_proj+v_proj）
- seed=42

唯一变量是骨干模型族（Qwen2 vs LLaMA2），用于检验"独立训练的异模型视角"
是否具备同构 self-debate 缺失的互补性。训练后用 eval_selective_hetero.py
在 dev 标定、test 终审。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

from debate_erc.config import LoRAConfig, TrainingConfig  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.models import AdapterManager, load_backbone  # noqa: E402
from debate_erc.training import SFTTrainer  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.train_critic_sft")

QWEN_DIR = "/home/lsy20252770/InstructERC/LLM_bases/DeepSeek-R1-Distill-Qwen-1.5B"
RAW_PATH = "/home/lsy20252770/Agent_Reason/PRC-Emo/data/raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="异构 critic（Qwen 系骨干）纯 SFT")
    parser.add_argument("--backbone", default=QWEN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("meld", "iemocap"), default="meld",
                        help="训练数据集（IEMOCAP 用 4 类 schema，同配方）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs/hetero_critic/qwen15b_sft_s42"))
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    schema = get_schema(args.dataset)
    train = load_split(args.dataset, RAW_PATH, "train", history_window=5)
    logger.info("%s train n=%d；骨干=%s", args.dataset.upper(), len(train), args.backbone)

    model, tokenizer = load_backbone(args.backbone)
    manager = AdapterManager(model, tokenizer)
    # 与 m0_sft 完全一致的 LoRA 规格
    manager.add("main", LoRAConfig(r=64, alpha=128, dropout=0.05,
                                   target_modules=("q_proj", "v_proj"), enabled=True))
    trainable, total, ratio = manager.trainable_params()
    logger.info("LoRA 可训练参数 %d/%d (%.2f%%)", trainable, total, ratio * 100)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model, tokenizer, schema, TrainingConfig(), out_dir / "checkpoints",
        adapter_manager=manager,
    )
    summary = trainer.train(train)
    manager.save("main", out_dir / "adapter_main")
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("训练完成，adapter 保存至 %s", out_dir / "adapter_main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
