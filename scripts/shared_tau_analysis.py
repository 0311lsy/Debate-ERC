#!/usr/bin/env python
"""共享阈值离线分析：解决单 seed dev 标定 τ 漂移问题。

合法口径：τ/margin 只从 dev 中产生，test 零调参。
两种方案：
  A. pooled-dev：三 seed dev 合并，选使平均 dev ΔW-F1 最大的单一 (τ, margin)，
     锁死应用于三 seed test；
  B. LOO：用另外两 seed 合并 dev 标定 τ，应用于留出 seed 的 test（无泄漏）。

test_records 仅在 p_a < τ_run 时收集了 y_b，因此 test 离线重算的 τ 不得超过该 seed
实际终审时的 τ_run（s42=0.70 / s43=0.50 / s44=0.60）。
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
sys.path.insert(0, "src")
from debate_erc.evaluation import compute_metrics
from debate_erc.utils.label_schema import MELD_SCHEMA

LABELS = list(MELD_SCHEMA.labels)
ROOT = Path("outputs/selective_hetero")
TAGS = ["qwen7b", "qwen7b_s43", "qwen7b_s44"]
TAUS = [round(0.4 + 0.05 * i, 2) for i in range(9)]   # 0.40..0.80
MARGINS = [0.0, 0.05, 0.10]


def load(tag, split):
    return [json.loads(l) for l in open(ROOT / tag / f"{split}_records.jsonl", encoding="utf-8")]


def final_pred(r, tau, margin):
    if (r.get("y_b") is not None and r["p_a"] < tau and r["y_b"] != r["y_a"]
            and r["p_b"] > r["p_a"] + margin):
        return r["y_b"]
    return r["y_a"]


def wf1(records, tau, margin):
    return compute_metrics([final_pred(r, tau, margin) for r in records],
                           [r["gold"] for r in records], LABELS)["wf1"]


def base_wf1(records):
    return compute_metrics([r["y_a"] for r in records],
                           [r["gold"] for r in records], LABELS)["wf1"]


dev = {t: load(t, "dev") for t in TAGS}
test = {t: load(t, "test") for t in TAGS}
dev_base = {t: base_wf1(dev[t]) for t in TAGS}
test_base = {t: base_wf1(test[t]) for t in TAGS}
tau_run = {"qwen7b": 0.70, "qwen7b_s43": 0.50, "qwen7b_s44": 0.60}

# ---------- A. pooled-dev 标定 ----------
pooled = [r for t in TAGS for r in dev[t]]
pooled_base = sum(dev_base.values()) / 3
grid = []
for tau, mg in itertools.product(TAUS, MARGINS):
    gain = sum(wf1(dev[t], tau, mg) - dev_base[t] for t in TAGS) / 3
    grid.append((gain, tau, mg))
grid.sort(reverse=True)
print("=== A. pooled-dev 网格 Top5（平均 dev ΔW-F1）===")
for g, tau, mg in grid[:5]:
    print(f"τ={tau:.2f} margin={mg:.2f}  mean dev Δ={g*100:+.3f}")
g_star, tau_star, mg_star = grid[0]
print(f"\n选定共享参数：τ*={tau_star} margin*={mg_star}（pooled dev Δ={g_star*100:+.3f}）")

print("\n=== 锁定共享参数后各 seed TEST（受 τ_run 覆盖限制）===")
ok = True
for t in TAGS:
    if tau_star > tau_run[t] + 1e-9:
        print(f"{t}: τ*={tau_star} > 该seed终审τ_run={tau_run[t]}，"
              f"test 缺 p_a∈({tau_run[t]},{tau_star}] 的 y_b，需补推理")
        ok = False
    else:
        d = wf1(test[t], tau_star, mg_star) - test_base[t]
        print(f"{t}: test {test_base[t]*100:.2f} → {wf1(test[t], tau_star, mg_star)*100:.2f}"
              f"（Δ {d*100:+.2f}）")
if ok:
    deltas = [wf1(test[t], tau_star, mg_star) - test_base[t] for t in TAGS]
    import statistics as st
    print(f"3-seed test ΔW-F1 = {st.mean(deltas)*100:+.2f} ± {st.stdev(deltas)*100:.2f}")

# ---------- B. LOO 标定 ----------
print("\n=== B. leave-one-seed-out：用另外两 seed dev 标定，应用于留出 seed test ===")
for i, held in enumerate(TAGS):
    others = [t for j, t in enumerate(TAGS) if j != i]
    best = None
    for tau, mg in itertools.product(TAUS, MARGINS):
        g = sum(wf1(dev[t], tau, mg) - dev_base[t] for t in others) / 2
        if best is None or g > best[0]:
            best = (g, tau, mg)
    _, tau_loo, mg_loo = best
    feasible = tau_loo <= tau_run[held] + 1e-9
    if feasible:
        d = wf1(test[held], tau_loo, mg_loo) - test_base[held]
        print(f"留出{held:10s}：LOO τ={tau_loo:.2f} m={mg_loo:.2f} → test Δ {d*100:+.2f}")
    else:
        print(f"留出{held:10s}：LOO τ={tau_loo:.2f} > τ_run={tau_run[held]}，需补推理")

# ---------- 参考：各 seed 独立 dev 最优 τ（诊断漂移）----------
print("\n=== 诊断：各 seed 独立 dev 最优点 vs 实际选用 ===")
for t in TAGS:
    best = max(itertools.product(TAUS, MARGINS),
               key=lambda x: wf1(dev[t], x[0], x[1]) - dev_base[t])
    print(f"{t:10s} dev最优 τ={best[0]:.2f}/m={best[1]:.2f}（dev Δ="
          f"{(wf1(dev[t],*best)-dev_base[t])*100:+.2f}）")
