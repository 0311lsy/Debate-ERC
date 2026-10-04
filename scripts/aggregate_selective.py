#!/usr/bin/env python
"""选择性辩论多 seed 聚合 + paired bootstrap 显著性（Layer 1 审稿终表口径）。

与 scripts/aggregate_results.py 分工：后者聚合 runner 正式 run 的 results.json；
本脚本聚合 eval_selective[_hetero].py 产物
（outputs/selective[_hetero]/<tag>/test_metrics.json + test_records.jsonl）。

用法：python scripts/aggregate_selective.py [--root outputs/selective_hetero]
          [--tags qwen7b qwen7b_s43 qwen7b_s44] [--n_boot 10000]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

try:
    from scipy.stats import binomtest
except ImportError:  # scipy 缺失时降级为无 McNemar
    binomtest = None

logger = get_logger("debate_erc.scripts.aggregate_selective")
LABELS = list(MELD_SCHEMA.labels)


def final_pred(r, tau, margin):
    """从记录复算最终预测（test_records 中未触发样本 y_b=None）。"""
    if (r["y_b"] is not None and r["p_a"] < tau and r["y_b"] != r["y_a"]
            and r["p_b"] > r["p_a"] + margin):
        return r["y_b"]
    return r["y_a"]


def paired_bootstrap(records, tau, margin, n_boot=10000, seed=2026):
    """ΔW-F1 / ΔMacro 的配对 bootstrap，返回各自 (mean, lo, hi)。"""
    rng = random.Random(seed)
    n = len(records)
    dw, dm = [], []
    for _ in range(n_boot):
        sub = [records[rng.randrange(n)] for _ in range(n)]
        golds = [r["gold"] for r in sub]
        mp = compute_metrics([r["y_a"] for r in sub], golds, LABELS)
        mf = compute_metrics([final_pred(r, tau, margin) for r in sub], golds, LABELS)
        dw.append(mf["wf1"] - mp["wf1"])
        dm.append(mf["macro_f1"] - mp["macro_f1"])

    def ci(vals):
        s = sorted(vals)
        return (sum(vals) / len(vals), s[int(0.025 * len(s))], s[int(0.975 * len(s))])

    return ci(dw), ci(dm)


def fmt(values):
    sd = stats.stdev(values) if len(values) > 1 else 0.0
    return f"{stats.mean(values)*100:.2f} ± {sd*100:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/selective_hetero")
    parser.add_argument("--tags", nargs="+",
                        default=["qwen7b", "qwen7b_s43", "qwen7b_s44"])
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()
    base = PROJECT_ROOT / args.root

    per_seed = []
    for tag in args.tags:
        d = base / tag
        metrics = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
        records = [json.loads(l) for l in open(d / "test_records.jsonl", encoding="utf-8")]
        tau = metrics["params_from_dev"]["tau"]
        margin = metrics["params_from_dev"]["margin"]
        ci_w, ci_m = paired_bootstrap(records, tau, margin, n_boot=args.n_boot)
        verdict = "显著为正" if ci_w[1] > 0 else ("显著为负" if ci_w[2] < 0 else "不显著")
        b, c = metrics["flip_correct"], metrics["flip_wrong"]
        mcn = None
        if binomtest is not None and b + c > 0:
            mcn = round(float(binomtest(max(b, c), b + c, 0.5,
                                        alternative="two-sided").pvalue), 4)
        per_seed.append({"tag": tag, "m": metrics, "ci_w": ci_w, "ci_m": ci_m,
                         "mcnemar_p": mcn})
        logger.info("[%s] ΔW-F1=%+.4f bootstrap95%%=[%+.4f, %+.4f] %s；"
                    "McNemar p=%s（改对%d/改错%d）",
                    tag, metrics["delta_wf1"], ci_w[1], ci_w[2], verdict,
                    mcn, b, c)

    pw = [s["m"]["proponent_only"]["wf1"] for s in per_seed]
    fw = [s["m"]["hetero_deliberation"]["wf1"] for s in per_seed]
    dw = [s["m"]["delta_wf1"] for s in per_seed]
    pm = [s["m"]["proponent_only"]["macro_f1"] for s in per_seed]
    fm = [s["m"]["hetero_deliberation"]["macro_f1"] for s in per_seed]
    pa = [s["m"]["proponent_only"]["accuracy"] for s in per_seed]
    fa = [s["m"]["hetero_deliberation"]["accuracy"] for s in per_seed]

    summary = {
        "n_seeds": len(per_seed), "root": args.root,
        "wf1": {"proponent": fmt(pw), "deliberation": fmt(fw), "delta": fmt(dw)},
        "macro_f1": {"proponent": fmt(pm), "deliberation": fmt(fm),
                     "delta": fmt([s["m"]["delta_macro"] for s in per_seed])},
        "accuracy": {"proponent": fmt(pa), "deliberation": fmt(fa)},
        "per_seed": [
            {"tag": s["tag"],
             "proponent_wf1": s["m"]["proponent_only"]["wf1"],
             "deliberation_wf1": s["m"]["hetero_deliberation"]["wf1"],
             "delta_wf1": s["m"]["delta_wf1"],
             "delta_wf1_boot95ci": [round(s["ci_w"][1], 4), round(s["ci_w"][2], 4)],
             "delta_macro_boot95ci": [round(s["ci_m"][1], 4), round(s["ci_m"][2], 4)],
             "trigger_rate": s["m"]["trigger_rate"], "n_flip": s["m"]["n_flip"],
             "flip_correct": s["m"]["flip_correct"], "flip_wrong": s["m"]["flip_wrong"], "mcnemar_p": s.get("mcnemar_p")}
            for s in per_seed],
    }
    out = base / "aggregate_3seeds.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {len(per_seed)}-seed 汇总（mean ± std，百分点）===")
    print(f"W-F1   {fmt(pw)} → {fmt(fw)}（Δ {fmt(dw)}）")
    print(f"Macro  {fmt(pm)} → {fmt(fm)}（Δ {summary['macro_f1']['delta']}）")
    print(f"Acc    {fmt(pa)} → {fmt(fa)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
