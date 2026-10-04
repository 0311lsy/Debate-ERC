#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SC 双温度 3-seed 聚合（M6 终审）。

数据源：
- outputs/self_consistency/sc_k5_s{42,43,44}/summary.json          (T=0.7)
- outputs/self_consistency/sc_k5_s{42,43,44}_t03/summary.json      (T=0.3)
贪心基线取 8-seed 主表固定 τ 口径的 Base W-F1：s42=68.60 / s43=68.97 / s44=68.76
（逐条 test_records 离线重算值，与主表 Table 1 一致）。

输出：
- 各温度/k 的 3-seed mean±std、配对 Δ、配对 t 检验
- 打印结果供回填文档 §7.1/§7.5 与 main.tex §5.2
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
# 与 Table 1 完全一致的贪心基线 W-F1（固定 τ=0.65 口径重算）
GREEDY = {42: 68.60, 43: 68.97, 44: 68.76}
DEBATE = {42: 69.63, 43: 68.70, 44: 69.66}  # 8-seed 主表 Ours 列


def load(seed: int, temp: float) -> dict:
    suffix = "" if temp == 0.7 else "_t03"
    p = ROOT / f"outputs/self_consistency/sc_k5_s{seed}{suffix}/summary.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def summarize(temp: float, key: str) -> None:
    """打印某温度/某 k 的 3-seed 聚合。"""
    sc_vals, deltas, debate_deltas = [], [], []
    print(f"\n--- T={temp}, {key.upper()} ---")
    print(f"{'seed':>5} {'greedy':>8} {'SC':>8} {'ΔSC':>8} {'debate':>8} {'Δdeb':>8}")
    for s in SEEDS:
        d = load(s, temp)
        sc = d[key]["wf1"] * 100
        g = GREEDY[s]
        sc_vals.append(sc)
        deltas.append(sc - g)
        debate_deltas.append(DEBATE[s] - g)
        print(f"{s:>5} {g:>8.2f} {sc:>8.2f} {sc-g:>+8.2f} {DEBATE[s]:>8.2f} {DEBATE[s]-g:>+8.2f}")
    sc_arr = np.array(sc_vals)
    d_arr = np.array(deltas)
    # 配对 t 检验（Δ vs 0）
    t_stat, p_val = stats.ttest_1samp(d_arr, 0.0)
    print(f"mean SC = {sc_arr.mean():.2f} ± {sc_arr.std(ddof=1):.2f}")
    print(f"mean Δ  = {d_arr.mean():+.2f} ± {d_arr.std(ddof=1):.2f}  "
          f"(paired t={t_stat:.2f}, p={p_val:.4f})")
    neg = int((d_arr < 0).sum())
    print(f"负 seed 数：{neg}/3")
    return sc_arr.mean(), d_arr.mean()


def main() -> None:
    print("=" * 60)
    print("SC 双温度 3-seed 聚合（MELD test, n=2610/seed）")
    print("=" * 60)
    g_mean = np.mean([GREEDY[s] for s in SEEDS])
    deb_mean = np.mean([DEBATE[s] for s in SEEDS])
    print(f"贪心基线 3-seed 均值：{g_mean:.2f}")
    print(f"异构辩论 3-seed 均值：{deb_mean:.2f}（Δ {deb_mean-g_mean:+.2f}）")

    rows = {}
    for temp in (0.7, 0.3):
        for key in ("sc3", "sc5"):
            rows[(temp, key)] = summarize(temp, key)

    print("\n" + "=" * 60)
    print("汇总（3-seed mean W-F1 / Δ vs greedy）")
    print("=" * 60)
    print(f"{'配置':<16} {'W-F1':>8} {'Δ':>8}")
    print(f"{'Greedy':<16} {g_mean:>8.2f} {'—':>8}")
    for temp in (0.7, 0.3):
        for key, label in (("sc3", "SC@3"), ("sc5", "SC@5")):
            m, dm = rows[(temp, key)]
            print(f"{label+f' T={temp}':<16} {m:>8.2f} {dm:>+8.2f}")
    print(f"{'Debate (ours)':<16} {deb_mean:>8.2f} {deb_mean-g_mean:>+8.2f}")


if __name__ == "__main__":
    main()
