#!/usr/bin/env python
"""Layer 0 配套：贪心生成口径评估（do_sample=False，可复现的标准生成式 ERC 口径）。

与 eval_logprob.py 的约束分类口径互为对照：
- eval_logprob.py：候选标签序列 logprob argmax（确定性、零生成、带校准概率）；
- 本脚本：模型自由生成 + 首行标签解析（标准生成式 ERC 评测口径，贪心解码）。

修复原 eval_ab_dpo.py 的两处测量缺陷：
1. adapter 加载：from_pretrained 命名加载 + 独立 adapter 名（同名 load 静默失效）；
2. 解码：do_sample=False、max_new_tokens=8（SFT 答案仅 " label\\n"，
   原 200 token 全部用于 prompt 复读，是推理耗时虚高的根源）。

用法：
    python scripts/eval_greedy.py \
        --checkpoints m0_sft=.../main m1_dpo=.../main [--n 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from debate_erc.agents.direct_agent import DirectClassifyAgent  # noqa: E402
from debate_erc.config import load_config  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.models import LLMGenerator, load_backbone  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.eval_greedy")


def main() -> int:
    parser = argparse.ArgumentParser(description="贪心生成口径评估")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/ablations/dpo_direct_classify.yaml"),
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=0, help="0=全量")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs/eval_greedy.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)
    test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
    if 0 < args.n < len(test):
        test = random.Random(args.seed).sample(test, args.n)
    golds = [s.gold_label for s in test]
    logger.info("test n=%d", len(test))

    ckpts = dict(item.partition("=")[::2] for item in args.checkpoints)
    model, tokenizer = load_backbone(cfg.backbone)
    first_name, first_dir = next(iter(ckpts.items()))
    model = PeftModel.from_pretrained(model, first_dir, adapter_name=first_name)
    for name, path in list(ckpts.items())[1:]:
        model.load_adapter(path, adapter_name=name)
    device = next(model.parameters()).device
    model.eval()

    gen = LLMGenerator(model, tokenizer)
    gen.candidate_labels = list(schema.labels)
    # 贪心 + 短生成（SFT 答案口径 " label\n"，首行解析；避免 200-token 复读）
    agent = DirectClassifyAgent(
        gen, schema,
        generation_kwargs={
            "do_sample": False, "max_new_tokens": 8,
            "temperature": 1.0, "top_p": 1.0,
        },
    )

    results = {}
    for name in ckpts:
        model.set_adapter(name)
        t0 = time.time()
        rows = []
        with torch.no_grad():
            for i, s in enumerate(test):
                out = agent.infer(s.context)
                rows.append({"idx": i, "gold": s.gold_label, "pred": out.label})
                if (i + 1) % 200 == 0:
                    logger.info("[%s] %d/%d", name, i + 1, len(test))
        elapsed = time.time() - t0
        preds = [r["pred"] for r in rows]
        m = compute_metrics(preds, golds, schema.labels)
        results[name] = {
            "wf1": m["wf1"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "per_class_f1": {lb: round(m["per_class_f1"].get(lb, 0.0), 4)
                             for lb in schema.labels},
            "pred_distribution": dict(Counter(preds)),
            "elapsed_sec": round(elapsed, 1),
        }
        logger.info(
            "[%s] W-F1=%.4f Macro=%.4f Acc=%.4f（%.0fs, %.2fs/条）",
            name, m["wf1"], m["macro_f1"], m["accuracy"],
            elapsed, elapsed / len(test),
        )
        with open(Path(args.out).with_suffix(f".{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("\n=== 贪心生成（全量 %d，do_sample=False）===" % len(test))
    for name, r in results.items():
        print("%-10s W-F1=%.4f Macro=%.4f Acc=%.4f" % (name, r["wf1"], r["macro_f1"], r["accuracy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
