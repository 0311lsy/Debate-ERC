#!/usr/bin/env python
"""论文主图多种子版（纯 CPU）：Fig.2 Pareto 与 Fig.3 risk-coverage。

与 make_paper_figures.py（s42 单种子）的差异：
  - Fig.2 自动探测所有同时具备 proponent test_records 与 critic 全量 dump 的种子，
    绘制 均值±1sd 带/误差棒（n = 可用种子数，标题中注明）。
  - SC@3/5 用 outputs/self_consistency/sc_k5_s{42,43,44} 的均值±sd。
  - Fig.3 的 three_branch curves 仅有 s42 数据，保持单种子并在标题注明。

critic s45–49 全量 dump 完成后重跑本脚本即升级为 8 种子版本。
输出 outputs/figures/fig_pareto.{png,pdf}（覆盖单种子版）与 pareto_points_multiseed.json。
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
MARGIN = 0.05
TAUS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]

# seed -> proponent pair 目录（critic dump 统一在 cross/critic{seed}_all_test.jsonl）
PROP_DIRS = {"42": "qwen7b"} | {str(s): f"qwen7b_s{s}_shared" for s in range(43, 50)}


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def available_seeds() -> list[str]:
    seeds = []
    for s, d in PROP_DIRS.items():
        if (HET / d / "test_records.jsonl").exists() and \
           (HET / "cross" / f"critic{s}_all_test.jsonl").exists():
            seeds.append(s)
    return seeds


def load_quads(seed: str):
    prop = {r["idx"]: r for r in read_jsonl(HET / PROP_DIRS[seed] / "test_records.jsonl")}
    crit = {r["idx"]: r for r in read_jsonl(HET / "cross" / f"critic{seed}_all_test.jsonl")}
    return [dict(prop[i], **{f"b_{k}": v for k, v in crit[i].items() if k in ("y_b", "p_b")})
            for i in sorted(prop) if i in crit]


def wf1(pred, rows):
    return compute_metrics(pred, [r["gold"] for r in rows], LABELS)["wf1"] * 100


def gated_preds(rows, tau):
    pred = []
    for r in rows:
        y = r["y_a"]
        if r["p_a"] < tau and r["b_y_b"] != r["y_a"] and r["b_p_b"] > r["p_a"] + MARGIN:
            y = r["b_y_b"]
        pred.append(y)
    return pred


def per_seed_curves(rows_by_seed: dict[str, list[dict]]):
    """每种子的 τ 扫描曲线与基线点，返回 {seed: {...}}。"""
    out = {}
    for s, rows in rows_by_seed.items():
        ya = [r["y_a"] for r in rows]
        yb = [r["b_y_b"] for r in rows]
        pa = np.array([r["p_a"] for r in rows])
        sweep = []
        for tau in TAUS:
            sweep.append({"tau": tau,
                          "cost": 1 + float((pa < tau).mean()),
                          "wf1": wf1(gated_preds(rows, tau), rows)})
        conf_pick = [r["y_a"] if r["p_a"] >= r["b_p_b"] else r["b_y_b"] for r in rows]
        full_flip = [r["b_y_b"] if (r["b_y_b"] != r["y_a"] and r["b_p_b"] > r["p_a"] + MARGIN)
                     else r["y_a"] for r in rows]
        oracle = [r["y_a"] if r["y_a"] == r["gold"] else r["b_y_b"] for r in rows]
        out[s] = {"sweep": sweep,
                  "greedy": wf1(ya, rows), "critic_only": wf1(yb, rows),
                  "conf_pick": wf1(conf_pick, rows), "always_flip": wf1(full_flip, rows),
                  "oracle": wf1(oracle, rows)}
    return out


def ms(vals):  # mean, sd（n=1 时 sd=0）
    vals = list(vals)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def fig_pareto(per_seed: dict[str, dict], seeds: list[str]):
    n = len(seeds)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)

    # Ours 曲线：逐 τ 聚合均值±sd
    costs = np.array([per_seed[seeds[0]]["sweep"][i]["cost"] for i in range(len(TAUS))])
    # cost 逐种子略有差异（触发率不同），用均值作 x
    cost_ms = [ms(per_seed[s]["sweep"][i]["cost"] for s in seeds) for i in range(len(TAUS))]
    xs = np.array([m for m, _ in cost_ms])
    ys_m = np.array([ms(per_seed[s]["sweep"][i]["wf1"] for s in seeds)[0] for i in range(len(TAUS))])
    ys_s = np.array([ms(per_seed[s]["sweep"][i]["wf1"] for s in seeds)[1] for i in range(len(TAUS))])
    ax.fill_between(xs, ys_m - ys_s, ys_m + ys_s, color="#1f6f8b", alpha=0.18, zorder=2)
    ax.plot(xs, ys_m, "-o", color="#1f6f8b", lw=2, ms=6, zorder=3,
            label="Ours: gated heterogeneous review")
    for x, y, tau in zip(xs, ys_m, TAUS):
        if tau in (0.40, 0.65, 0.90):
            ax.annotate(f"τ={tau}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5, color="#1f6f8b")

    def point(cost, key, marker, color, label, dx=0, dy=0):
        m, sd = ms(per_seed[s][key] for s in seeds)
        ax.errorbar([cost], [m], yerr=[sd], fmt="none", ecolor=color,
                    elinewidth=1.2, capsize=3, zorder=4, alpha=0.9)
        ax.scatter([cost], [m], marker=marker, s=110, color=color, zorder=5,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(label, (cost, m), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5)

    point(1.0, "greedy", "s", "#555555", "Greedy", dy=-15)
    point(2.0, "critic_only", "D", "#8e7cc3", "Critic-only", dx=10, dy=-10)
    point(2.0, "conf_pick", "^", "#e69138", "Confidence-pick", dx=-4, dy=-16)
    point(2.0, "always_flip", "v", "#cc4125", "Always-consult + flip", dx=10, dy=6)
    point(2.0, "oracle", "*", "#2e7d32", "Oracle upper bound", dx=10, dy=4)

    # Self-consistency 多种子、两温度（自动探测）
    sc = {"0.7": {3: [], 5: []}, "0.3": {3: [], 5: []}}
    for s in ("42", "43", "44"):
        for tag, suf in (("0.7", ""), ("0.3", "_t03")):
            p = PROJECT_ROOT / f"outputs/self_consistency/sc_k5_s{s}{suf}/summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                sc[tag][3].append(d["sc3"]["wf1"] * 100)
                sc[tag][5].append(d["sc5"]["wf1"] * 100)
    # T=0.7（默认，深灰）；T=0.3（浅灰，即使最好的 SC 变体仍不超过 greedy）
    for tag, col, tcol, fs, dy3 in (("0.7", "#999999", "#666666", 8.5, -14),
                                    ("0.3", "#c0c0c0", "#9a9a9a", 7.5, 10)):
        if not sc[tag][3]:
            continue
        for k, vals, dx, dy in ((3, sc[tag][3], -6, dy3),
                                (5, sc[tag][5], 6, 4 if tag == "0.7" else -12)):
            m, sd = ms(vals)
            ax.errorbar([k], [m], yerr=[sd], fmt="none", ecolor=col,
                        elinewidth=1.2, capsize=3, zorder=4)
            ax.scatter([k], [m], marker="X", s=110, color=col, zorder=5,
                       edgecolor="white")
            ax.annotate(f"SC@{k}" + (", $T{=}.3$" if tag == "0.3" else ""),
                        (k, m), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fs, color=tcol)

    # Homogeneous self-debate（s42 单种子，2x；outputs/selective 即同构 LLaMA2 双副本）
    sd_p = PROJECT_ROOT / "outputs/selective/test_metrics.json"
    if sd_p.exists():
        sd = json.loads(sd_p.read_text())
        y_sd = sd["selective_deliberation"]["wf1"] * 100
        ax.scatter([2.0], [y_sd], marker="P", s=130, color="#b45f06",
                   zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate("Self-debate (s1)", (2.0, y_sd),
                    textcoords="offset points", xytext=(8, -13),
                    fontsize=8.5, color="#7f4304", ha="left")

    # Weak 1.5B reviewer（s42 单种子；冻结门下仅 0.54% 触发，成本约 1.0x）
    w_p = PROJECT_ROOT / "outputs/selective_hetero/qwen15b/test_metrics.json"
    if w_p.exists():
        wd = json.loads(w_p.read_text())
        cost_w = 1.0 + wd["trigger_rate"] * (1.5 / 7.0)
        y_w = wd["hetero_deliberation"]["wf1"] * 100
        ax.scatter([cost_w], [y_w], marker="h", s=130, color="#a64d79",
                   zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate("Weak 1.5B (s1)", (cost_w, y_w),
                    textcoords="offset points", xytext=(10, 7),
                    fontsize=8.5, color="#7d335b")

    g_m = np.mean([per_seed[s]["greedy"] for s in seeds])
    ax.axhline(g_m, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Expected forward passes per utterance")
    ax.set_ylabel("Weighted F1 on MELD test (%)")
    ax.set_title(f"Cost–accuracy Pareto: heterogeneous review vs. baselines "
                 f"(mean ± sd, {n} seeds)")
    ax.set_xlim(0.65, 5.35)
    sc_all = sc["0.7"][3] + sc["0.7"][5] + sc["0.3"][3] + sc["0.3"][5]
    lo = min(ys_m.min(), g_m, y_sd if sd_p.exists() else 99,
             y_w if w_p.exists() else 99, min(sc_all or [99])) - 1.0
    hi = max(np.mean([per_seed[s]["oracle"] for s in seeds]), ys_m.max()) + 1.0
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pareto.{ext}")
    plt.close(fig)

    return {"n_seeds": n, "seeds": seeds,
            "tau_sweep_mean": [{"tau": t, "cost": round(float(x), 3),
                                "wf1_mean": round(float(m), 3), "wf1_sd": round(float(s), 3)}
                               for t, x, m, s in zip(TAUS, xs, ys_m, ys_s)],
            "points": {k: {"mean": round(ms(per_seed[s][k] for s in seeds)[0], 3),
                           "sd": round(ms(per_seed[s][k] for s in seeds)[1], 3)}
                       for k in ("greedy", "critic_only", "conf_pick", "always_flip", "oracle")},
            "sc": {tag: {k: [round(v, 3) for v in sc[tag][k]] for k in (3, 5)}
                   for tag in ("0.7", "0.3")},
            "single_seed": {"self_debate": {"cost": 2.0, "wf1": round(y_sd, 3)} if sd_p.exists() else None,
                            "weak_1p5b": {"cost": round(cost_w, 3), "wf1": round(y_w, 3)} if w_p.exists() else None},
            "per_seed": {s: per_seed[s] for s in seeds}}


def fig_risk_coverage():
    """three_branch 曲线仅有 s42 数据，保持单种子重绘（与旧脚本一致）。"""
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
    axin = ax.inset_axes([0.46, 0.57, 0.31, 0.35])
    axin.plot(d["A_abstain_only_coverage"], d["A_abstain_only_risk"], color="#4472c4", lw=1.6)
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
    seeds = available_seeds()
    print(f"可用种子（prop records + critic 全量 dump 齐备）: {seeds}")
    rows_by_seed = {s: load_quads(s) for s in seeds}
    for s, rows in rows_by_seed.items():
        print(f"  s{s}: n={len(rows)}")
    info = fig_pareto(per_seed_curves(rows_by_seed), seeds)
    fig_risk_coverage()
    (OUT / "pareto_points_multiseed.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "per_seed"}, indent=2))
    print(f"图已保存至 {OUT}/fig_pareto.{{png,pdf}}（{info['n_seeds']} 种子均值±sd）"
          f"与 fig_risk_coverage.{{png,pdf}}（s42）")


if __name__ == "__main__":
    main()
