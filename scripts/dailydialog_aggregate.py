#!/usr/bin/env python
"""DailyDialog 域外冻结迁移结果聚合（M2(a)）。

读取（eval_logprob 实际产物命名）：
  - outputs/dailydialog/greedy_proponent.json          → proponent 贪心 W-F1（参考）
  - outputs/dailydialog/logprob_proponent_details.jsonl → idx/gold/m0_sft_pred/m0_sft_p
  - outputs/dailydialog/logprob_critic_details.jsonl    → idx/gold/critic_pred/critic_p

门控基线取 logprob-argmax 口径（与 p_a 同一解码族；贪心 0.7913 仅作解码参考）。
复用主实验门控规则（tau=0.65, margin=0.05，冻结自主实验，不调参）
产出：
  - outputs/dailydialog/summary.json
  - 文档 §27 DailyDialog 域外结果
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.utils.label_schema import MELD_SCHEMA
from debate_erc.evaluation import compute_metrics


def main():
    base_dir = PROJECT_ROOT / "outputs/dailydialog"
    prop_log = base_dir / "logprob_proponent_details.jsonl"
    crit_log = base_dir / "logprob_critic_details.jsonl"
    greedy_json = base_dir / "greedy_proponent.json"

    if not prop_log.exists() or not crit_log.exists():
        print("DailyDialog 推理未完成，请等待队列", file=sys.stderr)
        return 1

    prop_rows = [json.loads(l) for l in prop_log.read_text().splitlines()]
    crit_rows = [json.loads(l) for l in crit_log.read_text().splitlines()]
    prop_by_idx = {r["idx"]: r for r in prop_rows}
    crit_by_idx = {r["idx"]: r for r in crit_rows}

    # 对齐
    common = sorted(set(prop_by_idx) & set(crit_by_idx))
    golds, base_preds, gated_preds = [], [], []
    tau, margin = 0.65, 0.05
    flips = {"rescue": 0, "harm": 0, "neutral": 0}
    low_conf = 0
    adopted = 0

    for idx in common:
        g = prop_by_idx[idx]["gold"]
        y_a = prop_by_idx[idx]["m0_sft_pred"]
        p_a = prop_by_idx[idx]["m0_sft_p"]
        y_b = crit_by_idx[idx]["critic_pred"]
        p_b = crit_by_idx[idx]["critic_p"]
        golds.append(g)
        base_preds.append(y_a)
        if p_a < tau:
            low_conf += 1
        if p_a >= tau:
            gated_preds.append(y_a)
        elif y_b != y_a and p_b > p_a + margin:
            gated_preds.append(y_b)
            adopted += 1
            if y_b == g and y_a != g:
                flips["rescue"] += 1
            elif y_b != g and y_a == g:
                flips["harm"] += 1
            else:
                flips["neutral"] += 1
        else:
            gated_preds.append(y_a)

    base_m = compute_metrics(base_preds, golds, MELD_SCHEMA.labels)
    gate_m = compute_metrics(gated_preds, golds, MELD_SCHEMA.labels)

    # 触发率：低置信即需 critic 一次前向（与主实验成本口径一致）
    trigger_rate = low_conf / len(common)
    greedy_ref = None
    if greedy_json.exists():
        g = json.loads(greedy_json.read_text()).get("m0_sft", {})
        greedy_ref = {k: g.get(k) for k in ("wf1", "macro_f1", "accuracy")}

    summary = {
        "n": len(common),
        "tau": tau, "margin": margin,
        "decoding": "logprob-argmax (base 与 p_a 同一解码族)",
        "proponent_greedy_ref": greedy_ref,
        "base_wf1": round(base_m["wf1"], 4),
        "base_macro_f1": round(base_m["macro_f1"], 4),
        "base_acc": round(base_m["accuracy"], 4),
        "gated_wf1": round(gate_m["wf1"], 4),
        "gated_macro_f1": round(gate_m["macro_f1"], 4),
        "gated_acc": round(gate_m["accuracy"], 4),
        "delta_wf1": round(gate_m["wf1"] - base_m["wf1"], 4),
        "trigger_rate_p_below_tau": round(trigger_rate, 4),
        "adoption_rate": round(adopted / len(common), 4),
        "expected_cost_x": round(1 + trigger_rate, 4),
        "flips": flips,
    }

    (base_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
