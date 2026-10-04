#!/usr/bin/env python
"""Self-Consistency（SC）对照基线：同预算推理时方法，论文比较集必含。

方法：对 m0_sft 主模型，同一 classify 提示采样 k 次（T=0.7 / top_p=0.9），
多数投票决定标签；平票时用候选标签序列 logsum 后验 argmax 破平
（score_labels 仅在平票时调用，避免每样本 +1 次前向）。
单次采样 5 票，取前 3 票报 SC@3、全部 5 票报 SC@5——两次共享同批样本，
比各跑一遍便宜且严格同分布。

对照口径（论文表格用）：
- 同一主模型（base_sft_s42）、同一 test 集、同一 compute_metrics；
- 贪心基线 y_a 直接复用 qwen7b_s42_shared/test_records.jsonl（同 adapter
  同贪心解码，y_a 逐条一致），避免重复 17 分钟贪心 pass；
- 预算对比：SC@3 = 3 次前向/条，SC@5 = 5 次前向/条；异构辩论期望
  1.27 次前向/条（贪心 1 + 触发率 27% × 1 次 critic）。

用法：
    python scripts/self_consistency.py --seed 42 --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from debate_erc.agents.prompt_builder import build_classify_prompt  # noqa: E402
from debate_erc.config import load_config  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.models import LLMGenerator, load_backbone  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402
from debate_erc.utils.post_process import parse_label  # noqa: E402

logger = get_logger("debate_erc.scripts.self_consistency")

OUT_DIR = PROJECT_ROOT / "outputs" / "self_consistency"
PROP_ADAPTER = PROJECT_ROOT / "outputs/runs/base_sft_s42/adapter_main/main"
BASE_RECORDS = (PROJECT_ROOT / "outputs/selective_hetero"
                / "qwen7b/test_records.jsonl")  # s42 主链终审落盘于 qwen7b/ 子目录
SAMPLE_KW = {"max_new_tokens": 8, "do_sample": True,
             "temperature": 0.7, "top_p": 0.9}


def vote(labels: list[str], dist: dict[str, float] | None) -> str:
    """多数投票；平票时用标签后验 argmax 破平。"""
    top = max(Counter(labels).values())
    tied = [lb for lb, c in Counter(labels).items() if c == top]
    if len(tied) == 1:
        return tied[0]
    if dist is None:  # pragma: no cover - 由调用方保证
        raise ValueError("平票必须提供 score_labels 分布")
    return max(tied, key=lambda lb: dist[lb])


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-Consistency 对照基线")
    parser.add_argument("--k", type=int, default=5, help="采样总票数（同时报 SC@3）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度；T=0.3 对照用于回应'SC 温度混淆'质疑")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prop_adapter", default=str(PROP_ADAPTER))
    parser.add_argument("--base-records", default=str(BASE_RECORDS),
                        help="同 seed 主模型贪心逐条记录（y_a）；多 seed 时必须按 seed 传入，"
                             "禁止跨 seed 复用")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/ablations/dpo_direct_classify.yaml"),
    )
    args = parser.parse_args()
    sample_kw = {"max_new_tokens": 8, "do_sample": True,
                 "temperature": args.temperature, "top_p": args.top_p}
    t_tag = "" if abs(args.temperature - 0.7) < 1e-6 else f"_t{int(args.temperature * 10):02d}"
    out_dir = OUT_DIR / f"sc_k{args.k}_s{args.seed}{t_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)
    labels = list(schema.labels)

    test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
    golds = [s.gold_label for s in test]

    # 贪心基线：复用同 seed 共享 τ 终审的逐条 y_a（按 idx 对齐，禁止跨 seed）
    base_by_idx: dict[int, dict] | None = None
    base_path = Path(args.base_records)
    if base_path.exists():
        rows = [json.loads(l) for l in base_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) == len(test):
            base_by_idx = {r["idx"]: r for r in rows}
            logger.info("贪心基线复用 %s（%d 条，按 idx 对齐）", base_path.name, len(rows))
        else:
            logger.warning("base records 条数 %d != test %d，不复用贪心基线", len(rows), len(test))
    logger.info("加载主模型：%s", args.prop_adapter)
    model, tok = load_backbone(cfg.backbone)
    model = PeftModel.from_pretrained(model, args.prop_adapter, adapter_name="m0_sft")
    model.set_adapter("m0_sft")
    model.eval()
    gen = LLMGenerator(model, tok)
    gen.candidate_labels = labels

    torch.manual_seed(args.seed)
    recs = []
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(test):
            prompt = build_classify_prompt(s.context, schema)
            votes = [parse_label(gen.generate(prompt, **sample_kw), schema)
                     for _ in range(args.k)]
            dist = None
            if len(set(votes[:3])) > 1 or len(set(votes)) > 1:
                dist = gen.score_labels(prompt, labels)  # 仅平票样本付出 1 次前向
            recs.append({"idx": i, "gold": s.gold_label,
                         "y_a": base_by_idx[i]["y_a"] if base_by_idx else None,
                         "votes": votes,
                         "sc3": vote(votes[:3], dist), "sc5": vote(votes, dist)})
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                logger.info("  %d/%d（%.2fs/条，剩余约 %.0fs）",
                            i + 1, len(test), rate, rate * (len(test) - i - 1))

    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    summary: dict = {"k": args.k, "seed": args.seed, "sampling": sample_kw,
                     "n_test": len(test)}
    for name, key in [("greedy", "y_a"), ("sc3", "sc3"), ("sc5", "sc5")]:
        if key == "y_a" and base_rows is None:
            continue
        m = compute_metrics([r[key] for r in recs], golds, labels)
        summary[name] = {"wf1": round(m["wf1"], 6), "macro_f1": round(m["macro_f1"], 6),
                         "accuracy": round(m["accuracy"], 6)}
    if "greedy" in summary:
        summary["delta_wf1_sc3_vs_greedy"] = round(
            summary["sc3"]["wf1"] - summary["greedy"]["wf1"], 6)
        summary["delta_wf1_sc5_vs_greedy"] = round(
            summary["sc5"]["wf1"] - summary["greedy"]["wf1"], 6)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    g = summary.get("greedy", {}).get("wf1", float("nan"))
    logger.info("SC 完成：greedy %.4f → SC@3 %.4f / SC@5 %.4f（前向预算 3x/5x vs 辩论 1.27x）",
                g, summary["sc3"]["wf1"], summary["sc5"]["wf1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
