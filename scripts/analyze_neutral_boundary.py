#!/usr/bin/env python
"""中性边界专项分析（纯 CPU，零新实验）。

回答"两 agent 中性级联"思路的三个前置问题：
  Q1 错误结构：情绪被吞 / neutral 误报 / 情绪间错分 各占多少（按种子）。
  Q2 跨模型互补：在 primary 的两类中性边界错误上，独立 critic 能否纠正、
     纠正率是否高于普通错误（决定"中性边界专家"有无独立信息）。
  Q3 三态软路由模拟：agent1=primary 给 confident neutral / confident emotion /
     不确定，仅"不确定"升级 critic；扫描双阈值，报成本-精度，与现有冻结门控
     (τ=0.65,m=0.05) 对比。同时模拟两种单向否决（neutral→emotion 与 emotion→
     neutral）的乐观（test 上选阈）上界。

输入：
  outputs/selective_hetero/{qwen7b,qwen7b_s43_shared,...}/test_records.jsonl
  outputs/selective_hetero/cross/critic{seed}_all_test.jsonl
自动探测两边都齐全的种子。
输出：outputs/review_followups/neutral_boundary_analysis.json + 控制台表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402

LABELS = list(MELD_SCHEMA.labels)
NEU = "neutral"
EMOS = [l for l in LABELS if l != NEU]
HET = PROJECT_ROOT / "outputs/selective_hetero"
PROP_DIRS = {"42": "qwen7b"} | {str(s): f"qwen7b_s{s}_shared" for s in range(43, 50)}
OUT = PROJECT_ROOT / "outputs/review_followups/neutral_boundary_analysis.json"
N_TEST = 2610


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def available_seeds() -> list[str]:
    seeds = []
    for s, d in PROP_DIRS.items():
        cp = HET / d / "test_records.jsonl"
        cc = HET / "cross" / f"critic{s}_all_test.jsonl"
        if cp.exists() and cc.exists():
            # 正在写入的 dump 不完整，跳过
            if sum(1 for _ in open(cc)) >= N_TEST:
                seeds.append(s)
    return sorted(seeds, key=int)


def load_pairs(seed: str):
    prop = {r["idx"]: r for r in read_jsonl(HET / PROP_DIRS[seed] / "test_records.jsonl")}
    crit = {r["idx"]: r for r in read_jsonl(HET / "cross" / f"critic{seed}_all_test.jsonl")}
    rows = []
    for i in sorted(prop):
        if i not in crit:
            continue
        a, b = prop[i], crit[i]
        rows.append({"idx": i, "gold": a["gold"], "ya": a["y_a"], "pa": a["p_a"],
                     "yb": b["y_b"], "pb": b["p_b"]})
    return rows


def wf1(pred, gold):
    return compute_metrics(pred, gold, tuple(LABELS))["wf1"] * 100


def analyze_seed(seed: str):
    rows = load_pairs(seed)
    gold = [r["gold"] for r in rows]
    ya = [r["ya"] for r in rows]
    n = len(rows)

    # ---- Q1 错误三分 ----
    swallowed = [r for r in rows if r["gold"] != NEU and r["ya"] == NEU]   # 情绪被吞
    falsealarm = [r for r in rows if r["gold"] == NEU and r["ya"] != NEU]  # neutral 误报
    emoconf = [r for r in rows if r["gold"] != NEU and r["ya"] != NEU and r["gold"] != r["ya"]]
    n_err = len(swallowed) + len(falsealarm) + len(emoconf)

    # ---- Q2 critic 在各类错误上的行为 ----
    def critic_behavior(sub, name):
        if not sub:
            return {"subset": name, "n": 0}
        agrees_err = sum(r["yb"] == r["ya"] for r in sub)          # 与错误一致
        corrects = sum(r["yb"] == r["gold"] for r in sub)          # critic 直接判对
        if name == "swallowed":
            predicts_neu = sum(r["yb"] == NEU for r in sub)
            return {"subset": name, "n": len(sub),
                    "critic_correct_pct": round(100 * corrects / len(sub), 1),
                    "critic_also_neutral_pct": round(100 * predicts_neu / len(sub), 1)}
        if name == "falsealarm":
            says_neu = sum(r["yb"] == NEU for r in sub)
            return {"subset": name, "n": len(sub),
                    "critic_correct_pct": round(100 * corrects / len(sub), 1),
                    "critic_says_neutral_pct": round(100 * says_neu / len(sub), 1)}
        return {"subset": name, "n": len(sub),
                "critic_correct_pct": round(100 * corrects / len(sub), 1),
                "critic_agrees_error_pct": round(100 * agrees_err / len(sub), 1)}

    # critic 总体正确率作参照
    critic_overall_acc = 100 * sum(r["yb"] == r["gold"] for r in rows) / n

    # 被吞样本按真实情绪分布
    sw_by_class = {e: sum(r["gold"] == e for r in swallowed) for e in EMOS}

    # ---- Q3a 单向否决乐观上界（test 选阈，明确标记为乐观）----
    # 方向1：primary 说 neutral，critic 高置信说情绪 → 改判情绪
    def veto_scan(subset, direction):
        best = None
        for th in [x / 100 for x in range(50, 100)]:
            pred = list(ya)
            flips = 0
            for r in subset:
                if direction == "neu2emo" and r["yb"] != NEU and r["pb"] >= th:
                    pred[r["idx"]] = r["yb"]; flips += 1
                if direction == "emo2neu" and r["yb"] == NEU and r["pb"] >= th:
                    pred[r["idx"]] = NEU; flips += 1
            s = wf1(pred, gold)
            if best is None or s > best[1]:
                best = (th, s, flips)
        return best

    # 两个方向各自的候选集
    cand_neu2emo = [r for r in rows if r["ya"] == NEU]   # primary neutral 才允许被翻成情绪
    cand_emo2neu = [r for r in rows if r["ya"] != NEU]
    b1 = veto_scan(cand_neu2emo, "neu2emo")
    b2 = veto_scan(cand_emo2neu, "emo2neu")

    # ---- Q3b 三态软路由 ----
    # confident neutral: ya==neu 且 pa>=tn → 直接 neutral
    # confident emotion: ya!=neu 且 pa>=te → 直接 ya
    # 其余升级 critic（用 yb）
    grid = []
    for tn in [x / 100 for x in range(50, 96, 5)]:
        for te in [x / 100 for x in range(50, 96, 5)]:
            pred = []
            esc = 0
            for r in rows:
                if r["ya"] == NEU and r["pa"] >= tn:
                    pred.append(NEU)
                elif r["ya"] != NEU and r["pa"] >= te:
                    pred.append(r["ya"])
                else:
                    pred.append(r["yb"]); esc += 1
            grid.append({"tn": tn, "te": te, "wf1": round(wf1(pred, gold), 3),
                         "escalate_pct": round(100 * esc / n, 1)})
    # 成本≈1+升级率（与正文同口径的 7B 次前向）
    best_grid = max(grid, key=lambda d: d["wf1"])
    # 成本不超过 1.30x 的最优
    cheap = [d for d in grid if 1 + d["escalate_pct"] / 100 <= 1.30]
    best_cheap = max(cheap, key=lambda d: d["wf1"]) if cheap else None

    base = wf1(ya, gold)
    # 现有门控结果（线上评估值，直接读 test_metrics 对齐）
    tm_path = HET / PROP_DIRS[seed] / "test_metrics.json"
    gate = json.loads(tm_path.read_text()) if tm_path.exists() else {}

    return {
        "seed": seed, "n": n, "base_wf1": round(base, 3),
        "base_acc": round(100 * sum(r["ya"] == r["gold"] for r in rows) / n, 2),
        "error_split": {"swallowed_emo_to_neu": len(swallowed),
                        "falsealarm_neu_to_emo": len(falsealarm),
                        "emo_to_emo": len(emoconf), "total": n_err,
                        "boundary_pct": round(100 * (len(swallowed) + len(falsealarm)) / n_err, 1)},
        "critic": {
            "overall_acc": round(critic_overall_acc, 2),
            "on_swallowed": critic_behavior(swallowed, "swallowed"),
            "on_falsealarm": critic_behavior(falsealarm, "falsealarm"),
            "on_emoconfusion": critic_behavior(emoconf, "emoconf"),
        },
        "swallowed_by_gold_class": sw_by_class,
        "veto_optimistic": {
            "neu2emo_best": {"th": b1[0], "wf1": round(b1[1], 3), "flips": b1[2],
                             "delta": round(b1[1] - base, 3)},
            "emo2neu_best": {"th": b2[0], "wf1": round(b2[1], 3), "flips": b2[2],
                             "delta": round(b2[1] - base, 3)},
        },
        "three_state_best": best_grid,
        "three_state_best_le130": best_cheap,
        "current_gate": {"wf1": round(gate.get("hetero_deliberation", {}).get("wf1", 0) * 100, 3),
                         "trigger_pct": round(gate.get("trigger_rate", 0) * 100, 1)},
    }


def main():
    seeds = available_seeds()
    print(f"可用种子（primary + critic 全量 dump 齐全）: {seeds}")
    results = [analyze_seed(s) for s in seeds]

    # 聚合
    def agg(key_path, fn="mean"):
        vals = []
        for r in results:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        m = sum(vals) / len(vals)
        return round(m, 3), round((sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5, 3)

    summary = {
        "n_seeds": len(seeds), "seeds": seeds,
        "base_wf1_mean_sd": agg(["base_wf1"]),
        "boundary_error_pct_mean_sd": agg(["error_split", "boundary_pct"]),
        "critic_correct_on_swallowed": agg(["critic", "on_swallowed", "critic_correct_pct"]),
        "critic_also_neutral_on_swallowed": agg(["critic", "on_swallowed", "critic_also_neutral_pct"]),
        "critic_correct_on_falsealarm": agg(["critic", "on_falsealarm", "critic_correct_pct"]),
        "critic_correct_on_emoconfusion": agg(["critic", "on_emoconfusion", "critic_correct_pct"]),
        "veto_neu2emo_delta": agg(["veto_optimistic", "neu2emo_best", "delta"]),
        "veto_emo2neu_delta": agg(["veto_optimistic", "emo2neu_best", "delta"]),
        "three_state_best_wf1": agg(["three_state_best", "wf1"]),
        "three_state_best_esc": agg(["three_state_best", "escalate_pct"]),
        "three_state_le130_wf1": agg(["three_state_best_le130", "wf1"]),
        "three_state_le130_esc": agg(["three_state_best_le130", "escalate_pct"]),
        "current_gate_wf1": agg(["current_gate", "wf1"]),
        "current_gate_trigger": agg(["current_gate", "trigger_pct"]),
        "per_seed": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 控制台 ----
    print("\n=== Q1 错误三分（均值）===")
    es0 = results[0]["error_split"]
    print(f"  情绪被吞 {es0['swallowed_emo_to_neu']} / neutral误报 {es0['falsealarm_neu_to_emo']} / "
          f"情绪间 {es0['emo_to_emo']}（s{seeds[0]}，各类种子几乎相同）")
    print(f"  跨中性边界错误占比 {summary['boundary_error_pct_mean_sd'][0]}%")
    print("\n=== Q2 独立 critic 在错误子集上的纠正率 ===")
    print(f"  critic 总体 acc {results[0]['critic']['overall_acc']}%（s{seeds[0]}）")
    for r in results[:1]:
        print(f"  被吞(情绪→neutral) {r['critic']['on_swallowed']['n']}个: "
              f"critic判对 {r['critic']['on_swallowed']['critic_correct_pct']}%, "
              f"也压成neutral {r['critic']['on_swallowed']['critic_also_neutral_pct']}%")
        print(f"  误报(neutral→情绪) {r['critic']['on_falsealarm']['n']}个: "
              f"critic判对 {r['critic']['on_falsealarm']['critic_correct_pct']}%")
        print(f"  情绪间错分 {r['critic']['on_emoconfusion']['n']}个: "
              f"critic判对 {r['critic']['on_emoconfusion']['critic_correct_pct']}%")
        print(f"  被吞样本真实类分布: {r['swallowed_by_gold_class']}")
    print(f"\n  多种子均值: 被吞上纠正率 {summary['critic_correct_on_swallowed'][0]}%, "
          f"误报上 {summary['critic_correct_on_falsealarm'][0]}%, "
          f"情绪间错分上 {summary['critic_correct_on_emoconfusion'][0]}%")
    print(f"  被吞样本 critic 同样压成 neutral: {summary['critic_also_neutral_on_swallowed'][0]}%")
    print("\n=== Q3a 单向否决（test 选阈=乐观上界）===")
    print(f"  neu→emo 否决 ΔW-F1 {summary['veto_neu2emo_delta'][0]}±{summary['veto_neu2emo_delta'][1]}")
    print(f"  emo→neu 否决 ΔW-F1 {summary['veto_emo2neu_delta'][0]}±{summary['veto_emo2neu_delta'][1]}")
    print("\n=== Q3b 三态软路由 ===")
    print(f"  无约束最优: W-F1 {summary['three_state_best_wf1'][0]}±{summary['three_state_best_wf1'][1]} "
          f"@ 升级 {summary['three_state_best_esc'][0]}% (≈{1+summary['three_state_best_esc'][0]/100:.2f}×)")
    print(f"  ≤1.30×:    W-F1 {summary['three_state_le130_wf1'][0]}±{summary['three_state_le130_wf1'][1]} "
          f"@ 升级 {summary['three_state_le130_esc'][0]}%")
    print(f"  对照 primary {summary['base_wf1_mean_sd'][0]}，现有冻结门控 "
          f"{summary['current_gate_wf1'][0]} @ {summary['current_gate_trigger'][0]}%")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
