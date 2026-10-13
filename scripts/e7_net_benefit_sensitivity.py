#!/usr/bin/env python
"""E7：部署决策净收益敏感性分析（纯 CPU）。

从主表 8 种子池化 test_records 实测条件概率出发：
  φ = P(trigger), e = P(error|trigger), r = P(rescue|error,trigger), h = P(harm|correct,trigger)
净修正 = N·φ·(e·r − (1−e)·h)（对称代价）；
非对称代价 λ = harm 代价 / rescue 收益：净效用 = N·φ·(e·r − λ(1−e)·h)。
扫描 φ、e、λ，输出 json + 双面板图。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SH = PROJECT_ROOT / "outputs/selective_hetero"
HINT_DIRS = {42: "qwen7b", 43: "qwen7b_s43", 44: "qwen7b_s44",
             **{s: f"qwen7b_s{s}_shared" for s in range(45, 50)}}
TAU, MARGIN = 0.65, 0.05
OUT_JSON = PROJECT_ROOT / "outputs/review_followups/e7_net_benefit_sensitivity.json"
OUT_FIG = PROJECT_ROOT / "outputs/figures/fig_net_benefit.pdf"


def main() -> None:
    # 权威口径：aggregate_8seeds_uniform（原始全量 records 重放）。
    # 注意：s42-44 的 test_records 只含各自 dev-τ 以下的 critic 提案
    # （0.7/0.5/0.6），按冻结 0.65 重放会漏计 [τ_dev,0.65) 区间，
    # 因此 e/r/h 一律取 aggregate 池化值（与论文一致），不用当前 records。
    agg = json.loads((SH / "aggregate_8seeds_uniform.json").read_text())
    tau, margin = agg["tau"], agg["margin"]
    rows = agg["rows"]
    N = sum(r["n"] for r in rows)
    trig = sum(round(r["consult"] * r["n"]) for r in rows)
    R = sum(r["rc"] for r in rows)
    H = sum(r["rw"] for r in rows)
    e = 0.5653  # 论文口径（aggregate 原始数据），见上注
    trig_err = round(trig * e)
    phi = trig / N
    r = R / trig_err
    h = H / (trig - trig_err)

    phi_grid = np.round(np.arange(0.20, 0.351, 0.01), 3)
    e_grid = np.round(np.arange(0.45, 0.701, 0.025), 3)
    lam_grid = [0.5, 1.0, 2.0, 5.0]

    # 面板1：净修正/10^5 随 φ（固定实测 e,r,h）
    net_per_1e5 = {float(p): round(1e5 * p * (e * r - (1 - e) * h), 1) for p in phi_grid}
    # 面板2：盈亏平衡 e*(λ) 与净效用曲线（N=1e5, φ 实测）
    breakeven = {lam: round(lam * h / (r + lam * h), 4) for lam in lam_grid}
    util_grid = {lam: [round(1e5 * phi * (ee * r - lam * (1 - ee) * h), 1) for ee in e_grid]
                 for lam in lam_grid}

    out = {
        "measured_pooled_8seed": {"n": N, "trigger_rate": round(phi, 4),
                                  "triggered": trig,
                                  "triggered_errors": trig_err,
                                  "e_error_given_trigger": round(e, 4),
                                  "r_rescue_given_error": round(r, 4),
                                  "h_harm_given_correct": round(h, 4),
                                  "net_rate": round((R - H) / N, 5),
                                  "net_per_1e5_measured": round(1e5 * (R - H) / N, 1)},
        "net_corrections_per_1e5_vs_trigger_rate": net_per_1e5,
        "breakeven_e_by_cost_ratio": breakeven,
        "net_utility_per_1e5_vs_e_grid": util_grid,
        "e_grid": e_grid.tolist(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ps = sorted(net_per_1e5)
    ax.plot(ps, [net_per_1e5[p] for p in ps], lw=2)
    ax.axvline(phi, ls="--", c="gray", lw=1)
    ax.annotate(f"deployed gate\n$\\varphi$={phi:.3f}", (phi, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 10), fontsize=8)
    ax.set_xlabel("trigger rate $\\varphi$")
    ax.set_ylabel("net corrected / $10^5$ utterances")
    ax.set_title("(a) Net corrections vs. trigger rate", fontsize=9)

    ax = axes[1]
    for lam in lam_grid:
        ax.plot(e_grid, util_grid[lam], lw=1.6,
                label=f"$\\lambda$={lam:g} (break-even $e^*$={breakeven[lam]:.2f})")
    ax.axvline(e, ls="--", c="gray", lw=1)
    ax.annotate("measured\n$e$=0.56", (e, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("$e=P(\\mathrm{error}\\mid\\mathrm{trigger})$")
    ax.set_ylabel("net utility / $10^5$")
    ax.set_title("(b) Cost-asymmetry sensitivity", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(json.dumps(out["measured_pooled_8seed"], indent=1))
    print("breakeven:", breakeven)


if __name__ == "__main__":
    main()
