#!/usr/bin/env python
"""论文主图生成（纯 CPU）：Fig.2 Pareto 与 Fig.3 三分支 risk-coverage。

全部 s42 同口径（统一 τ/margin 离线重算），输出 outputs/figures/：
  fig_pareto.png/pdf        成本（期望前向/样本）vs W-F1
  fig_risk_coverage.png/pdf 覆盖度-风险曲线（仅弃权 / 弃权+辩论 / oracle）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402

LABELS = list(MELD_SCHEMA.labels)
HET = PROJECT_ROOT / "outputs/selective_hetero"
OUT = PROJECT_ROOT / "outputs/figures"
TAU_REF, MARGIN = 0.65, 0.05


def load_quads():
    prop = {json.loads(l)["idx"]: json.loads(l)
            for l in (HET / "qwen7b/test_records.jsonl").read_text(encoding="utf-8").splitlines()}
    crit = {json.loads(l)["idx"]: json.loads(l)
            for l in (HET / "cross/critic42_all_test.jsonl").read_text(encoding="utf-8").splitlines()}
    return [dict(prop[i], **{f"b_{k}": v for k, v in crit[i].items() if k in ("y_b", "p_b")})
            for i in sorted(prop) if i in crit]


def wf1(pred, rows):
    return compute_metrics(pred, [r["gold"] for r in rows], LABELS)["wf1"]


def fig_pareto(rows):
    gold = [r["gold"] for r in rows]
    ya = [r["y_a"] for r in rows]
    yb = [r["b_y_b"] for r in rows]
    pa = np.array([r["p_a"] for r in rows])
    pb = np.array([r["b_p_b"] for r in rows])

    # τ 扫描：期望成本 = 1 + P(p_a<τ)
    taus = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
    xs, ys, tags = [], [], []
    for tau in taus:
        pred = []
        for r in rows:
            y = r["y_a"]
            if r["p_a"] < tau and r["b_y_b"] != r["y_a"] and r["b_p_b"] > r["p_a"] + MARGIN:
                y = r["b_y_b"]
            pred.append(y)
        call = float((pa < tau).mean())
        xs.append(1 + call)
        ys.append(wf1(pred, rows) * 100)
        tags.append(f"τ={tau}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    ax.plot(xs, ys, "-o", color="#1f6f8b", lw=2, ms=6, zorder=3,
            label="Ours: gated heterogeneous review")
    # 仅标注代表性 τ，交错偏移避免重叠
    label_offsets = {0.40: (0, 9), 0.50: (-2, -15), 0.65: (0, 10), 0.90: (0, 9)}
    for x, y, t in zip(xs, ys, tags):
        tv = float(t.split("=")[1])
        if tv in label_offsets:
            dx, dy = label_offsets[tv]
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha="center", fontsize=7.5, color="#1f6f8b")

    def point(cost, pred, marker, color, label, dx=0, dy=0):
        ax.scatter([cost], [wf1(pred, rows) * 100], marker=marker, s=110,
                   color=color, zorder=4, edgecolor="white", linewidth=0.8)
        ax.annotate(label, (cost, wf1(pred, rows) * 100),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8.5)

    point(1.0, ya, "s", "#555555", "Greedy", dy=-15)
    # 全量 critic 策略（成本 2.0）
    point(2.0, yb, "D", "#8e7cc3", "Critic-only", dx=10, dy=-10)
    conf_pick = [r["y_a"] if r["p_a"] >= r["b_p_b"] else r["b_y_b"] for r in rows]
    point(2.0, conf_pick, "^", "#e69138", "Confidence-pick", dx=-4, dy=-16)
    full_flip = [r["b_y_b"] if (r["b_y_b"] != r["y_a"] and r["b_p_b"] > r["p_a"] + MARGIN)
                 else r["y_a"] for r in rows]
    point(2.0, full_flip, "v", "#cc4125", "Always-consult + flip", dx=10, dy=6)
    oracle = [r["y_a"] if r["y_a"] == r["gold"] else r["b_y_b"] for r in rows]
    point(2.0, oracle, "*", "#2e7d32", "Oracle upper bound", dx=10, dy=4)

    # Self-consistency（summary.json）
    sc = json.loads((PROJECT_ROOT / "outputs/self_consistency/sc_k5_s42/summary.json").read_text())
    ax.scatter([3, 5], [sc["sc3"]["wf1"] * 100, sc["sc5"]["wf1"] * 100],
               marker="X", s=110, color="#999999", zorder=4, edgecolor="white")
    ax.annotate("SC@3", (3, sc["sc3"]["wf1"] * 100), textcoords="offset points",
                xytext=(-6, -14), fontsize=8.5, color="#666666")
    ax.annotate("SC@5", (5, sc["sc5"]["wf1"] * 100), textcoords="offset points",
                xytext=(6, 4), fontsize=8.5, color="#666666")

    ax.axhline(wf1(ya, rows) * 100, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Expected forward passes per utterance")
    ax.set_ylabel("Weighted F1 on MELD test (%)")
    ax.set_title("Cost–accuracy Pareto: heterogeneous review vs. baselines (seed 42)")
    ax.set_xlim(0.65, 5.35)
    ax.set_ylim(66.0, 75.9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pareto.{ext}")
    plt.close(fig)

    return {"tau_sweep": [{"tau": t, "cost": round(x, 3), "wf1": round(y, 3)}
                          for t, x, y in zip(taus, xs, ys)],
            "points": {"greedy": round(wf1(ya, rows) * 100, 2),
                       "critic_only": round(wf1(yb, rows) * 100, 2),
                       "conf_pick": round(wf1(conf_pick, rows) * 100, 2),
                       "always_flip": round(wf1(full_flip, rows) * 100, 2),
                       "oracle": round(wf1(oracle, rows) * 100, 2),
                       "sc3": round(sc["sc3"]["wf1"] * 100, 2),
                       "sc5": round(sc["sc5"]["wf1"] * 100, 2)}}


def fig_risk_coverage():
    d = np.load(OUT.parent / "three_branch/curves.npz")
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    ax.plot(d["A_abstain_only_coverage"], d["A_abstain_only_risk"],
            color="#4472c4", lw=2, label="Selective (abstain only)")
    ax.plot(d["DA_debate_plus_abstain_coverage"], d["DA_debate_plus_abstain_risk"],
            color="#c00000", lw=2, label="Three-branch: debate + abstain")
    ax.plot(d["oracle_A_cov"], d["oracle_A_risk"], color="#4472c4", lw=1.2, ls="--",
            label="Oracle (abstain only)")
    ax.plot(d["oracle_DA_cov"], d["oracle_DA_risk"], color="#c00000", lw=1.2, ls="--",
            label="Oracle (debate + abstain)")
    ax.set_xlabel("Coverage (fraction of utterances answered)")
    ax.set_ylabel("Selective risk (error rate on answered)")
    ax.set_title("Risk–coverage on MELD test (seed 42, τ=0.65)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)

    # 插图：放大高覆盖区——辩论的贡献集中在"不能大量弃权"的部署区间
    # 顶部居中（白底遮挡少量主曲线，x 起点 0.44 与左上图例水平不重叠）
    axin = ax.inset_axes([0.46, 0.57, 0.31, 0.35])
    axin.plot(d["A_abstain_only_coverage"], d["A_abstain_only_risk"],
              color="#4472c4", lw=1.6)
    axin.plot(d["DA_debate_plus_abstain_coverage"], d["DA_debate_plus_abstain_risk"],
              color="#c00000", lw=1.6)
    axin.set_xlim(0.70, 1.0)
    axin.set_ylim(0.18, 0.315)
    axin.grid(alpha=0.25)
    axin.tick_params(labelsize=7)
    axin.text(0.04, 0.93, "high-coverage zoom", transform=axin.transAxes,
              fontsize=7.5, va="top", ha="left",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.indicate_inset_zoom(axin, edgecolor="black", lw=0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_risk_coverage.{ext}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_quads()
    info = fig_pareto(rows)
    fig_risk_coverage()
    (OUT / "pareto_points.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"图已保存至 {OUT}/fig_pareto.{{png,pdf}} 与 fig_risk_coverage.{{png,pdf}}")


if __name__ == "__main__":
    main()
