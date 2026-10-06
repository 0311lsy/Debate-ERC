#!/usr/bin/env python
"""审稿跟进：纯 CPU 补充实验批次（不占用 GPU）。

Exp-1: InstructERC 固定 proponent × critic s42/43/44（R1 冻结 / R2 双阈值dev校准 / R3 margin-τ重校准）
Exp-2: R2 双阈值扩展 τ_a 网格（至 0.9995）+ top5/邻域平台稳定性
Exp-3: 主管线 s42/43/44 的 τ×margin 敏感性矩阵（全量 critic 重放）

所有门控均为离线记录合并，不加载任何模型。
用法: python scripts/review_cpu_followups.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SH = PROJECT_ROOT / "outputs" / "selective_hetero"
IE_DIR = PROJECT_ROOT / "outputs" / "instructerc_gated_v2"
OUT = PROJECT_ROOT / "outputs" / "review_followups"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"]
SEEDS = [42, 43, 44]
CRITIC_DIR = {42: "qwen7b", 43: "qwen7b_s43", 44: "qwen7b_s44"}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def wf1(golds, preds):
    return f1_score(golds, preds, average="weighted", labels=LABELS)


def merge(prop_rows, critic_rows):
    """按 idx inner join，返回含 gold/y_a/p_a/y_b/p_b 的记录。"""
    c = {r["idx"]: r for r in critic_rows}
    out = []
    for r in prop_rows:
        cr = c.get(r["idx"])
        if cr is not None:
            out.append({"idx": r["idx"], "gold": r["gold"],
                        "y_a": r["y_a"], "p_a": r["p_a"],
                        "y_b": cr["y_b"], "p_b": cr["p_b"]})
    return out


def margin_rule(rows, tau, margin):
    preds, trig, flips, rescue, harm = [], 0, 0, 0, 0
    for r in rows:
        flip = r["p_a"] < tau and r["y_b"] != r["y_a"] and r["p_b"] > r["p_a"] + margin
        trig += int(r["p_a"] < tau)
        flips += int(flip)
        if flip:
            rescue += int(r["y_b"] == r["gold"] and r["y_a"] != r["gold"])
            harm += int(r["y_b"] != r["gold"] and r["y_a"] == r["gold"])
        preds.append(r["y_b"] if flip else r["y_a"])
    return preds, {"trigger_rate": trig / len(rows), "n_flip": flips,
                   "rescue": rescue, "harm": harm}


def two_threshold(rows, tau_a, tau_b):
    preds, trig, flips, rescue, harm = [], 0, 0, 0, 0
    for r in rows:
        flip = r["p_a"] < tau_a and r["y_b"] != r["y_a"] and r["p_b"] > tau_b
        trig += int(r["p_a"] < tau_a)
        flips += int(flip)
        if flip:
            rescue += int(r["y_b"] == r["gold"] and r["y_a"] != r["gold"])
            harm += int(r["y_b"] != r["gold"] and r["y_a"] == r["gold"])
        preds.append(r["y_b"] if flip else r["y_a"])
    return preds, {"trigger_rate": trig / len(rows), "n_flip": flips,
                   "rescue": rescue, "harm": harm}


def pack(golds, preds, st, base):
    w = wf1(golds, preds)
    return {"wf1": round(w * 100, 2), "delta": round((w - base) * 100, 2),
            "trigger_rate": round(st["trigger_rate"], 4), "n_flip": st["n_flip"],
            "rescue": st["rescue"], "harm": st["harm"]}


def msd(vals):
    return {"mean": round(statistics.mean(vals), 3),
            "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "values": [round(v, 2) for v in vals]}


# ── 载入 InstructERC proponent + 三个 critic ──────────────────────────
ie_test = load_jsonl(IE_DIR / "test_records.jsonl")
ie_dev = load_jsonl(IE_DIR / "dev_records.jsonl")

ie_merged_test, ie_merged_dev = {}, {}
for s in SEEDS:
    ct = load_jsonl(SH / "cross" / f"critic{s}_all_test.jsonl")
    cd = load_jsonl(SH / CRITIC_DIR[s] / "dev_records.jsonl")
    ie_merged_test[s] = merge(ie_test, ct)
    ie_merged_dev[s] = merge(ie_dev, cd)
    assert len(ie_merged_test[s]) == 2610, f"seed{s} test merge {len(ie_merged_test[s])}"
    assert len(ie_merged_dev[s]) == 1109, f"seed{s} dev merge {len(ie_merged_dev[s])}"


# ═══════════════ Exp-1 + Exp-2 ═══════════════
TAU_A_EXT = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999, 0.9995]
TAU_B_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
TAU_R3_GRID = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999, 0.9995]

exp1 = {"proponent": "InstructERC ep8 (W-F1 66.47 test / 64.51 dev)",
        "critic_seeds": SEEDS, "per_seed": {}}
exp2 = {"tau_a_grid_extended": TAU_A_EXT, "per_seed": {}}

for s in SEEDS:
    test, dev = ie_merged_test[s], ie_merged_dev[s]
    g_t, g_d = [r["gold"] for r in test], [r["gold"] for r in dev]
    base_t, base_d = wf1(g_t, [r["y_a"] for r in test]), wf1(g_d, [r["y_a"] for r in dev])

    # R1 冻结
    p, st = margin_rule(test, 0.65, 0.05)
    r1 = pack(g_t, p, st, base_t)

    # R3 margin 规则，τ 在 dev 重校准（扩展网格）
    r3_grid = []
    for tau in TAU_R3_GRID:
        pd_, _ = margin_rule(dev, tau, 0.05)
        r3_grid.append({"tau": tau, "dev_wf1": round(wf1(g_d, pd_) * 100, 2)})
    best_tau = max(r3_grid, key=lambda x: x["dev_wf1"])["tau"]
    p, st = margin_rule(test, best_tau, 0.05)
    r3 = pack(g_t, p, st, base_t)
    r3["tau_dev_selected"] = best_tau

    # R2 双阈值，dev 网格校准（扩展网格）
    grid = []
    for ta in TAU_A_EXT:
        for tb in TAU_B_GRID:
            pd_, std_ = two_threshold(dev, ta, tb)
            grid.append({"tau_a": ta, "tau_b": tb,
                         "dev_wf1": wf1(g_d, pd_) * 100,
                         "dev_trigger": std_["trigger_rate"]})
    grid_sorted = sorted(grid, key=lambda x: (x["dev_wf1"], -x["dev_trigger"]), reverse=True)
    best = grid_sorted[0]
    p, st = two_threshold(test, best["tau_a"], best["tau_b"])
    r2 = pack(g_t, p, st, base_t)
    r2["tau_a_dev"] = best["tau_a"]
    r2["tau_b_dev"] = best["tau_b"]
    r2["dev_wf1"] = round(best["dev_wf1"], 2)

    exp1["per_seed"][str(s)] = {"base_test_wf1": round(base_t * 100, 2),
                                 "base_dev_wf1": round(base_d * 100, 2),
                                 "R1_frozen_margin": r1,
                                 "R2_two_threshold_devcal": r2,
                                 "R3_margin_dev_tau": r3}

    # Exp-2：top5 + 邻域 test Δ（邻域=tau_a 档位 ±1，固定 tau_b=best.tau_b）
    top5 = grid_sorted[:5]
    idx_best = TAU_A_EXT.index(best["tau_a"])
    neigh = []
    for j in (idx_best - 1, idx_best, idx_best + 1):
        if 0 <= j < len(TAU_A_EXT):
            ta = TAU_A_EXT[j]
            p, st = two_threshold(test, ta, best["tau_b"])
            neigh.append({"tau_a": ta, "tau_b": best["tau_b"],
                          "test_delta": round((wf1(g_t, p) - base_t) * 100, 2),
                          "trigger_rate": round(st["trigger_rate"], 4),
                          "rescue": st["rescue"], "harm": st["harm"]})
    exp2["per_seed"][str(s)] = {
        "dev_selected": {"tau_a": best["tau_a"], "tau_b": best["tau_b"],
                         "dev_wf1": round(best["dev_wf1"], 2)},
        "at_grid_boundary": best["tau_a"] == TAU_A_EXT[-1],
        "dev_top5": [{"tau_a": g["tau_a"], "tau_b": g["tau_b"],
                      "dev_wf1": round(g["dev_wf1"], 2)} for g in top5],
        "test_neighborhood": neigh,
    }

# 汇总
for rule in ["R1_frozen_margin", "R2_two_threshold_devcal", "R3_margin_dev_tau"]:
    deltas = [exp1["per_seed"][str(s)][rule]["delta"] for s in SEEDS]
    trig = [exp1["per_seed"][str(s)][rule]["trigger_rate"] for s in SEEDS]
    exp1.setdefault("summary", {})[rule] = {
        "delta_wf1": msd(deltas), "trigger_rate": msd(trig)}
exp1["summary"]["rescue_harm_total"] = {
    rule: {"rescue": sum(exp1["per_seed"][str(s)][rule]["rescue"] for s in SEEDS),
           "harm": sum(exp1["per_seed"][str(s)][rule]["harm"] for s in SEEDS)}
    for rule in ["R1_frozen_margin", "R2_two_threshold_devcal", "R3_margin_dev_tau"]}

neigh_deltas = [v["test_neighborhood"] for v in exp2["per_seed"].values()]
flat = [x["test_delta"] for nb in neigh_deltas for x in nb]
exp2["neighborhood_delta_range"] = {"min": min(flat), "max": max(flat),
                                    "all_positive": all(x > 0 for x in flat)}

(OUT / "exp1_instructerc_3seeds.json").write_text(json.dumps(exp1, indent=2))
(OUT / "exp2_grid_extension.json").write_text(json.dumps(exp2, indent=2))


# ═══════════════ Exp-3：主管线 τ×m 敏感性 ═══════════════
TAUS = [0.50, 0.60, 0.65, 0.70, 0.80]
MARGINS = [0.0, 0.025, 0.05, 0.10, 0.15]

exp3 = {"seeds": SEEDS, "taus": TAUS, "margins": MARGINS,
        "note": "主管线 LLaMA2 proponent + Qwen critic；3 种子平均 ΔW-F1（百分点）",
        "mean_delta": [], "mean_trigger": [], "per_seed_delta": {}}

bases = {}
for s in SEEDS:
    prop = load_jsonl(SH / CRITIC_DIR[s] / "test_records.jsonl")
    crit = load_jsonl(SH / "cross" / f"critic{s}_all_test.jsonl")
    rows = merge(prop, crit)
    assert len(rows) == 2610
    golds = [r["gold"] for r in rows]
    base = wf1(golds, [r["y_a"] for r in rows])
    mat_d, mat_t = [], []
    for tau in TAUS:
        row_d, row_t = [], []
        for m in MARGINS:
            preds, st = margin_rule(rows, tau, m)
            row_d.append(round((wf1(golds, preds) - base) * 100, 2))
            row_t.append(round(st["trigger_rate"], 3))
        mat_d.append(row_d)
        mat_t.append(row_t)
    exp3["per_seed_delta"][str(s)] = mat_d
    bases[s] = (golds, base, mat_t)

# 跨种子平均
for i, tau in enumerate(TAUS):
    exp3["mean_delta"].append([
        round(statistics.mean([exp3["per_seed_delta"][str(s)][i][j] for s in SEEDS]), 2)
        for j in range(len(MARGINS))])
    exp3["mean_trigger"].append([
        round(statistics.mean([bases[s][2][i][j] for s in SEEDS]), 3)
        for j in range(len(MARGINS))])

# 部署点与"全为正"统计
i0, j0 = TAUS.index(0.65), MARGINS.index(0.05)
exp3["deploy_point_0.65_0.05"] = {
    "mean_delta": exp3["mean_delta"][i0][j0],
    "per_seed_delta": [exp3["per_seed_delta"][str(s)][i0][j0] for s in SEEDS]}
allpos = sum(1 for row in exp3["mean_delta"] for v in row if v > 0)
exp3["cells_positive"] = f"{allpos}/{len(TAUS)*len(MARGINS)}"
exp3["base_wf1_per_seed"] = {str(s): round(bases[s][1] * 100, 2) for s in SEEDS}

(OUT / "exp3_margin_sensitivity.json").write_text(json.dumps(exp3, indent=2))


# ═══════════════ 控制台摘要 ═══════════════
print("=" * 70)
print("Exp-1  InstructERC × 3 critic seeds（ΔW-F1 百分点）")
for s in SEEDS:
    r = exp1["per_seed"][str(s)]
    print(f"  s{s}: base {r['base_test_wf1']:.2f} | R1 {r['R1_frozen_margin']['delta']:+.2f} "
          f"| R2 {r['R2_two_threshold_devcal']['delta']:+.2f} "
          f"(τa={r['R2_two_threshold_devcal']['tau_a_dev']},τb={r['R2_two_threshold_devcal']['tau_b_dev']}) "
          f"| R3 {r['R3_margin_dev_tau']['delta']:+.2f} (τ={r['R3_margin_dev_tau']['tau_dev_selected']})")
for rule, v in exp1["summary"].items():
    if "delta_wf1" in v:
        print(f"  {rule}: {v['delta_wf1']['mean']:+.2f} ± {v['delta_wf1']['sd']:.2f}  "
              f"trigger {v['trigger_rate']['mean']:.3f}")
print("  R/H total:", exp1["summary"]["rescue_harm_total"])
print("=" * 70)
print("Exp-2  扩展网格：边界命中 =",
      {s: exp2["per_seed"][s]["at_grid_boundary"] for s in exp2["per_seed"]})
print("  邻域 test Δ 范围:", exp2["neighborhood_delta_range"])
for s, v in exp2["per_seed"].items():
    print(f"  s{s} selected τa={v['dev_selected']['tau_a']} top5 τa="
          f"{[g['tau_a'] for g in v['dev_top5']]}")
print("=" * 70)
print("Exp-3  主管线 τ×m 平均 ΔW-F1（行 τ, 列 m=", MARGINS, "）")
for i, tau in enumerate(TAUS):
    print(f"  τ={tau:<4} {exp3['mean_delta'][i]}")
print("  正收益格子:", exp3["cells_positive"], "| 部署点:", exp3["deploy_point_0.65_0.05"])
print(f"\n全部结果 → {OUT}")
