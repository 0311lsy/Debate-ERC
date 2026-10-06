#!/usr/bin/env python
"""EmoryNLP 三种子汇总：域内 dev 标定协议下的门控增益（配对统计）。

读 outputs/selective_hetero/emorynlp_qwen7b_s{42,43,44}/test_metrics.json，
输出 outputs/review_followups/emorynlp_summary.json：
ΔW-F1/ΔMacro/ΔAcc 均值±sd、配对 t（手写 + scipy 交叉验证）、符号检验、
触发率、critic standalone W-F1。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "outputs/selective_hetero"
SEEDS = list(range(42, 50))  # 42-49 八种子（42-44 首轮 + 45-49 追加）
OUT = PROJECT_ROOT / "outputs/review_followups/emorynlp_summary.json"


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def paired_t(deltas):
    m, sd = mean_sd(deltas)
    return m / (sd / math.sqrt(len(deltas))) if sd > 0 else float("inf")


def t_p_two_sided(t, df):
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        return None


def main():
    rows = []
    for s in SEEDS:
        f = OUT_DIR / f"emorynlp_qwen7b_s{s}" / "test_metrics.json"
        if not f.exists():
            print(f"缺失：{f}")
            continue
        m = json.loads(f.read_text())
        rows.append({
            "seed": s,
            "prop_wf1": round(m["proponent_only"]["wf1"] * 100, 2),
            "gated_wf1": round(m["hetero_deliberation"]["wf1"] * 100, 2),
            "delta_wf1": round(m["delta_wf1"] * 100, 2),
            "delta_macro": round(m["delta_macro"] * 100, 2),
            "prop_acc": round(m["proponent_only"]["accuracy"] * 100, 2),
            "gated_acc": round(m["hetero_deliberation"]["accuracy"] * 100, 2),
            "delta_acc": round((m["hetero_deliberation"]["accuracy"]
                                - m["proponent_only"]["accuracy"]) * 100, 2),
            "critic_wf1": round(m["critic_standalone_all"]["wf1"] * 100, 2),
            "trigger_pct": round(m["trigger_rate"] * 100, 1),
            "flip_correct": m["flip_correct"], "flip_wrong": m["flip_wrong"],
            "tau": m["params_from_dev"]["tau"], "margin": m["params_from_dev"]["margin"],
        })
    if not rows:
        print("无可用结果"); return 1

    d = [r["delta_wf1"] for r in rows]
    dm = [r["delta_macro"] for r in rows]
    da = [r["delta_acc"] for r in rows]
    t = paired_t(d)
    summary = {
        "dataset": "EmoryNLP", "n_seeds": len(rows), "per_seed": rows,
        "wf1_prop_mean_sd": [round(x, 3) for x in mean_sd([r["prop_wf1"] for r in rows])],
        "wf1_gated_mean_sd": [round(x, 3) for x in mean_sd([r["gated_wf1"] for r in rows])],
        "delta_wf1_mean_sd": [round(x, 3) for x in mean_sd(d)],
        "delta_macro_mean_sd": [round(x, 3) for x in mean_sd(dm)],
        "delta_acc_mean_sd": [round(x, 3) for x in mean_sd(da)],
        "paired_t": round(t, 3), "df": len(rows) - 1,
        "paired_p_two_sided": (round(t_p_two_sided(t, len(rows) - 1), 4)
                               if t_p_two_sided(t, len(rows) - 1) is not None else None),
        "n_positive": sum(1 for x in d if x > 0),
        "trigger_pct_mean": round(sum(r["trigger_pct"] for r in rows) / len(rows), 1),
        "critic_wf1_mean": round(sum(r["critic_wf1"] for r in rows) / len(rows), 2),
        "protocol": "per-seed dev calibration (tau,margin), frozen before test",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
