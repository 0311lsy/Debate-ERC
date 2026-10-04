#!/usr/bin/env python
"""采纳规则离线搜索（严格 dev→test）。

规则族（一次锁定，test 只评一次）：
  翻案当且仅当  p_a < τ  且  p_b ≥ c（critic 置信度地板）
                 且  y_a ≠ y_b  且  p_b > p_a + margin
选择标准：pooled-dev（3327 条）平均 ΔW-F1 最大，
         附加硬约束——三个 seed 的 dev ΔW-F1 全部 ≥ 0（排除 s43 型单 seed 出血）。
test 覆盖限制：s42 记录覆盖 p_a<0.70；s43/s44(shared) 覆盖 p_a<0.65，
         故统一规则 τ ≤ 0.65（超出则需补推理，本脚本标注不可行）。
"""
from __future__ import annotations
import json, sys, itertools, statistics as st
from pathlib import Path
sys.path.insert(0, "src")
from debate_erc.evaluation import compute_metrics
from debate_erc.utils.label_schema import MELD_SCHEMA

L = list(MELD_SCHEMA.labels)
ROOT = Path("outputs/selective_hetero")
TAGS = ["qwen7b", "qwen7b_s43", "qwen7b_s44"]
TEST_PATHS = {"qwen7b": "qwen7b/test_records.jsonl",
              "qwen7b_s43": "qwen7b_s43_shared/test_records.jsonl",
              "qwen7b_s44": "qwen7b_s44_shared/test_records.jsonl"}


def load(tag, split):
    p = ROOT / tag / f"{split}_records.jsonl"
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def final_pred(r, tau, margin, c):
    if (r.get("y_b") is not None and r["p_a"] < tau and r["p_b"] >= c
            and r["y_b"] != r["y_a"] and r["p_b"] > r["p_a"] + margin):
        return r["y_b"]
    return r["y_a"]


def wf1(rs, tau, margin, c):
    return compute_metrics([final_pred(r, tau, margin, c) for r in rs],
                           [r["gold"] for r in rs], L)["wf1"]


def flipcounts(rs, tau, margin, c):
    fc = fw = 0
    for r in rs:
        y = final_pred(r, tau, margin, c)
        if y != r["y_a"]:
            fc += (y == r["gold"]); fw += (y != r["gold"])
    return fc, fw


dev = {t: load(t, "dev") for t in TAGS}
test = {t: load(t, Path(TEST_PATHS[t]).parts[0].split("/")[0] and
                str(Path(TEST_PATHS[t])).split("/", 1)[1].replace("_records.jsonl", "").replace("test", "test"))
        for t in TAGS}
test = {t: [json.loads(x) for x in open(ROOT / TEST_PATHS[t], encoding="utf-8")] for t in TAGS}
dev_base = {t: wf1(dev[t], -1, 9, 9) for t in TAGS}   # 不可能触发 → proponent-only
test_base = {t: wf1(test[t], -1, 9, 9) for t in TAGS}

TAUS = [round(0.40 + 0.05 * i, 2) for i in range(6)]   # 0.40..0.65（test 覆盖内）
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]
CS = [0.0, 0.5, 0.6, 0.7, 0.8]

rows = []
for tau, m, c in itertools.product(TAUS, MARGINS, CS):
    per_dev = [(wf1(dev[t], tau, m, c) - dev_base[t]) * 100 for t in TAGS]
    pooled = sum(per_dev) / 3
    rows.append((pooled, min(per_dev), tau, m, c, per_dev))

# 约束：三 seed dev 全 ≥ 0，再按 pooled 排序
feasible = [r for r in rows if r[1] >= 0.0]
feasible.sort(reverse=True)
print("=== pooled-dev Top8（已约束每 seed dev Δ≥0）===")
print(f"{'pooled':>7} {'worst':>6} {'τ':>5} {'margin':>6} {'p_b≥':>5} | 三seed dev Δ")
for pooled, worst, tau, m, c, per in feasible[:8]:
    print(f"{pooled:+6.3f} {worst:+6.2f} {tau:5.2f} {m:6.2f} {c:5.2f} | "
          f"{per[0]:+.2f} {per[1]:+.2f} {per[2]:+.2f}")

print("\n=== 不加约束的 pooled-dev Top3（对照：可能牺牲单 seed）===")
for pooled, worst, tau, m, c, per in sorted(rows, reverse=True)[:3]:
    print(f"{pooled:+6.3f} {worst:+6.2f} τ={tau:.2f} m={m:.2f} c={c:.2f} | {per}")

# 锁定约束下的最优规则，test 只评这一个
pooled, worst, tau, m, c, per = feasible[0]
print(f"\n=== 锁定规则 τ={tau} margin={m} critic p_b≥{c} → TEST 一次终审 ===")
dtest = []
for t in TAGS:
    b, f = test_base[t], wf1(test[t], tau, m, c)
    fc, fw = flipcounts(test[t], tau, m, c)
    d = (f - b) * 100; dtest.append(d)
    print(f"{t:10s}: {b*100:.2f} → {f*100:.2f}（Δ {d:+.2f}）翻转对{fc}/错{fw}")
print(f"3-seed test ΔW-F1 = {st.mean(dtest):+.2f} ± {st.stdev(dtest):.2f}")
import scipy.stats as ss
tt, pp = ss.ttest_1samp(dtest, 0)
print(f"t={tt:.2f} p={pp:.3f}；三 seed 符号：{'/'.join('+' if x>0 else '−' for x in dtest)}")
