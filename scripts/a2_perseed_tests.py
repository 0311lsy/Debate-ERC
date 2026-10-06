#!/usr/bin/env python
"""A2: 逐种子统计检验（纯 CPU，只读 test_records.jsonl，不加载模型）。

口径（与论文预注册一致）：
- baseline 预测 = y_a（proponent only）
- ours 预测重放冻结门控 τ=0.65 / margin=0.05：
  p_a<0.65 且 y_b 非空且 y_b!=y_a 且 p_b>p_a+0.05 → y_b，否则 y_a
- 逐种子 McNemar（精确二项，b/c 为不一致对）；8 种子符号检验（精确）；
  paired t / Wilcoxon 作为对照（与正文一致）。

输出: outputs/review_followups/a2_perseed_tests.json + 终端可读表。
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "outputs" / "selective_hetero"
TAU, MARGIN = 0.65, 0.05
SEED_DIRS = {"42": "qwen7b"} | {str(s): f"qwen7b_s{s}_shared" for s in range(43, 50)}
OUT = ROOT / "outputs" / "review_followups" / "a2_perseed_tests.json"


def replay_ours(r):
    yb, pb = r.get("y_b"), r.get("p_b")
    if r["p_a"] < TAU and yb is not None and yb != r["y_a"] and pb > r["p_a"] + MARGIN:
        return yb
    return r["y_a"]


rows_out = []
deltas = []
for s, d in SEED_DIRS.items():
    recs = [json.loads(x) for x in open(SH / d / "test_records.jsonl", encoding="utf-8")]
    b = c = 0  # McNemar: b=baseline对ours错, c=baseline错ours对
    base_correct = gated_correct = 0
    for r in recs:
        yb_pred = r["y_a"]
        yo_pred = replay_ours(r)
        b_ok = yb_pred == r["gold"]
        o_ok = yo_pred == r["gold"]
        base_correct += b_ok
        gated_correct += o_ok
        if b_ok and not o_ok:
            b += 1
        elif o_ok and not b_ok:
            c += 1
    n_disc = b + c
    p_mcnemar = binomtest(c, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    delta_wf1 = None  # W-F1 delta 由 aggregate 提供，这里只算 acc 口径的 delta
    deltas.append((gated_correct - base_correct) / len(recs) * 100)
    rows_out.append({
        "seed": int(s), "n": len(recs),
        "mcnemar_b_baseline_correct_only": b,
        "mcnemar_c_ours_correct_only": c,
        "mcnemar_exact_p": round(p_mcnemar, 4),
        "acc_delta_pp": round((gated_correct - base_correct) / len(recs) * 100, 3),
    })

n_pos = sum(1 for d in deltas if d > 0)
sign_p = binomtest(n_pos, len(deltas), 0.5).pvalue
t_stat = statistics.mean(deltas) / (statistics.stdev(deltas) / math.sqrt(len(deltas)))
wil = wilcoxon(deltas)

result = {
    "protocol": f"frozen gate tau={TAU}, margin={MARGIN}; ours replayed from test_records",
    "per_seed": rows_out,
    "sign_test": {"n_positive": n_pos, "n": len(deltas), "exact_p_two_sided": round(sign_p, 4)},
    "paired_t_on_acc_delta": round(t_stat, 3),
    "wilcoxon": {"statistic": float(wil.statistic), "p": round(float(wil.pvalue), 4)},
    "acc_delta_mean_sd": [round(statistics.mean(deltas), 3), round(statistics.stdev(deltas), 3)],
    "note": "pooled McNemar across seeds is invalid (same 2610 utterances repeated); per-seed only.",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"{'seed':>5}{'n':>6}{'b(prop-only)':>14}{'c(ours-only)':>14}{'McNemar p':>11}{'Δacc(pp)':>10}")
for r in rows_out:
    print(f"{r['seed']:>5}{r['n']:>6}{r['mcnemar_b_baseline_correct_only']:>14}"
          f"{r['mcnemar_c_ours_correct_only']:>14}{r['mcnemar_exact_p']:>11.4f}{r['acc_delta_pp']:>10.3f}")
print(f"\n符号检验: {n_pos}/{len(deltas)} 为正, exact p(two-sided)={sign_p:.4f}")
print(f"paired t(acc Δ)={t_stat:.3f}  Wilcoxon p={wil.pvalue:.4f}")
print(f"acc Δ = {statistics.mean(deltas):.3f} ± {statistics.stdev(deltas):.3f} pp")
print(f"已保存 {OUT}")
