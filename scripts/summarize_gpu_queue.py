#!/usr/bin/env python
"""GPU 队列全部完成后的统一汇总（纯 CPU，只读产物 json，不加载模型）。

三套统计：
  1) 角色互换：Qwen prop + LLaMA2 critic，8 种子（cross_qwen_prop_s{42..49}）
  2) hint 消融：带 hint 主管线 8 种子 vs nohint 8 种子（逐种子配对）
  3) B5 数据独立性：半数据 critic（qwen7b_s42_halfdata）vs 全量 s42
另附：主方向 8 种子作为角色互换对照基准（aggregate_8seeds_uniform.json）。

用法: python scripts/summarize_gpu_queue.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SH = PROJECT_ROOT / "outputs" / "selective_hetero"
SEEDS = list(range(42, 50))
OUT = PROJECT_ROOT / "outputs" / "review_followups" / "gpu_queue_summary.json"


def load(p):
    return json.load(open(p))


def tstat(vals):
    n = len(vals)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return m, sd, t


def paired_t(diffs):
    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return m, sd, t


def cohen_dz(diffs):
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return statistics.mean(diffs) / sd if sd > 0 else float("inf")


result = {}

# ── 1. 主方向（对照基准）────────────────────────────────────────────
agg = load(SH / "aggregate_8seeds_uniform.json")
main_rows = {r["seed"]: r for r in agg["rows"]}
main_d = [main_rows[s]["delta"] * 100 for s in SEEDS]
m, sd, t = tstat(main_d)
result["main_direction_LLaMAprop_Qwencritic"] = {
    "seeds": SEEDS,
    "delta_wf1_per_seed": [round(x, 2) for x in main_d],
    "mean": round(m, 3), "sd": round(sd, 3), "t": round(t, 3),
    "n_positive": sum(x > 0 for x in main_d),
    "base_mean": round(statistics.mean([main_rows[s]["base"] * 100 for s in SEEDS]), 2),
    "gated_mean": round(statistics.mean([main_rows[s]["debate"] * 100 for s in SEEDS]), 2),
    "confirmatory_s45_49": {
        "per_seed": [round(main_rows[s]["delta"] * 100, 2) for s in range(45, 50)],
        "note": "未参与 pooled-dev 阈值校准"},
}

# ── 2. 角色互换 8 种子 ──────────────────────────────────────────────
cross = []
missing_cross = []
for s in SEEDS:
    f = SH / f"cross_qwen_prop_s{s}" / "test_metrics.json"
    if f.exists():
        m_ = load(f)
        cross.append({
            "seed": s,
            "base": round(m_["proponent_only"]["wf1"] * 100, 2),
            "gated": round(m_["hetero_deliberation"]["wf1"] * 100, 2),
            "delta": round(m_["delta_wf1"] * 100, 2),
            "trigger_rate": round(m_["trigger_rate"], 4),
            "rescue": m_["flip_correct"], "harm": m_["flip_wrong"],
        })
    else:
        missing_cross.append(s)

if len(cross) == 8:
    deltas = [r["delta"] for r in cross]
    m, sd, t = tstat(deltas)
    # 与主方向的配对差异（同种子）
    pair_diff = [c - m_ for c, m_ in zip(deltas, main_d)]
    pm, psd, pt = paired_t(pair_diff)
    result["role_reversal_Qwenprop_LLaMAcritic"] = {
        "per_seed": cross,
        "base_mean": round(statistics.mean([r["base"] for r in cross]), 2),
        "gated_mean": round(statistics.mean([r["gated"] for r in cross]), 2),
        "delta_mean": round(m, 3), "delta_sd": round(sd, 3), "t": round(t, 3),
        "n_positive": sum(x > 0 for x in deltas),
        "cohen_dz": round(cohen_dz(deltas), 2),
        "rescue_total": sum(r["rescue"] for r in cross),
        "harm_total": sum(r["harm"] for r in cross),
        "trigger_mean": round(statistics.mean([r["trigger_rate"] for r in cross]), 4),
        "vs_main_direction": {
            "per_seed_delta_diff": [round(x, 2) for x in pair_diff],
            "mean_diff": round(pm, 3), "t_paired": round(pt, 3),
            "interpretation": "差值≈0 则支持角色互换对称性"},
    }
else:
    result["role_reversal_Qwenprop_LLaMAcritic"] = {"status": "incomplete", "missing": missing_cross}

# ── 3. hint vs nohint 8 种子配对 ────────────────────────────────────
hint_dirs = {42: "qwen7b", 43: "qwen7b_s43", 44: "qwen7b_s44",
             **{s: f"qwen7b_s{s}_shared" for s in range(45, 50)}}
hint_rows, nohint_rows, missing_hint = [], [], []
for s in SEEDS:
    fh = SH / hint_dirs[s] / "test_metrics.json"
    fn = SH / f"qwen7b_s{s}_nohint" / "test_metrics.json"
    if fh.exists():
        mm = load(fh)
        hint_rows.append({"seed": s,
                          "base": round(mm["proponent_only"]["wf1"] * 100, 2),
                          "gated": round(mm["hetero_deliberation"]["wf1"] * 100, 2),
                          "delta": round(mm["delta_wf1"] * 100, 2)})
    if fn.exists():
        mm = load(fn)
        nohint_rows.append({"seed": s,
                            "base": round(mm["proponent_only"]["wf1"] * 100, 2),
                            "gated": round(mm["hetero_deliberation"]["wf1"] * 100, 2),
                            "delta": round(mm["delta_wf1"] * 100, 2)})

if len(nohint_rows) == 8 and len(hint_rows) == 8:
    hd = {r["seed"]: r["delta"] for r in hint_rows}
    nd = {r["seed"]: r["delta"] for r in nohint_rows}
    pair = [nd[s] - hd[s] for s in SEEDS]  # nohint - hint
    m1, sd1, t1 = tstat([hd[s] for s in SEEDS])
    m2, sd2, t2 = tstat([nd[s] for s in SEEDS])
    pm, psd, pt = paired_t(pair)
    result["hint_vs_nohint"] = {
        "hint_delta_per_seed": [hd[s] for s in SEEDS],
        "nohint_delta_per_seed": [nd[s] for s in SEEDS],
        "hint_mean_sd_t": [round(m1, 3), round(sd1, 3), round(t1, 3)],
        "nohint_mean_sd_t": [round(m2, 3), round(sd2, 3), round(t2, 3)],
        "nohint_minus_hint_per_seed": [round(x, 2) for x in pair],
        "paired_mean_diff": round(pm, 3), "paired_t": round(pt, 3),
        "cohen_dz": round(cohen_dz(pair), 2),
        "nohint_better_seeds": [s for s in SEEDS if nd[s] > hd[s]],
        "decision_note": "配对差显著>0 才支持换主表；否则保留预注册 hint 主表并改写消融段",
    }
else:
    result["hint_vs_nohint"] = {
        "status": "incomplete",
        "hint_done": [r["seed"] for r in hint_rows],
        "nohint_done": [r["seed"] for r in nohint_rows],
        "nohint_preview": nohint_rows,
    }

# ── 4. B5 半数据 critic ─────────────────────────────────────────────
f5 = SH / "qwen7b_s42_halfdata" / "test_metrics.json"
f5train = PROJECT_ROOT / "outputs/hetero_critic/qwen7b_sft_s42_halfdata/train_summary.json"
if f5.exists():
    mm = load(f5)
    full = load(SH / "qwen7b" / "test_metrics.json")
    result["B5_halfdata_critic"] = {
        "train_summary": load(f5train) if f5train.exists() else None,
        "half_delta": round(mm["delta_wf1"] * 100, 2),
        "half_gated_wf1": round(mm["hetero_deliberation"]["wf1"] * 100, 2),
        "half_base_wf1": round(mm["proponent_only"]["wf1"] * 100, 2),
        "half_trigger": round(mm["trigger_rate"], 4),
        "full_delta_s42": round(full["delta_wf1"] * 100, 2),
        "half_minus_full": round((mm["delta_wf1"] - full["delta_wf1"]) * 100, 2),
        "note": "半数据与全量同 proponent(s42)，直接可比；单种子探索性对照"}
else:
    result["B5_halfdata_critic"] = {"status": "pending"}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))

# ── 控制台报告 ──────────────────────────────────────────────────────
def show(title, key, fields):
    print("=" * 72)
    print(title)
    v = result.get(key, {})
    if v.get("status") in ("incomplete", "pending"):
        print("  [未完成]", v)
        return
    for f in fields:
        if f in v:
            print(f"  {f}: {v[f]}")

print("主方向 Δ:", result["main_direction_LLaMAprop_Qwencritic"]["delta_wf1_per_seed"],
      "mean", result["main_direction_LLaMAprop_Qwencritic"]["mean"])
show("角色互换 8 种子", "role_reversal_Qwenprop_LLaMAcritic",
     ["per_seed", "base_mean", "gated_mean", "delta_mean", "delta_sd", "t",
      "n_positive", "cohen_dz", "rescue_total", "harm_total", "trigger_mean",
      "vs_main_direction"])
show("hint vs nohint", "hint_vs_nohint",
     ["hint_delta_per_seed", "nohint_delta_per_seed", "hint_mean_sd_t",
      "nohint_mean_sd_t", "nohint_minus_hint_per_seed", "paired_mean_diff",
      "paired_t", "cohen_dz", "nohint_better_seeds"])
show("B5 半数据", "B5_halfdata_critic",
     ["train_summary", "half_delta", "full_delta_s42", "half_minus_full",
      "half_trigger"])
print("=" * 72)
print("结果 →", OUT)
