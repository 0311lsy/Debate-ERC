#!/usr/bin/env python
"""E1：触发桶内零成本启发式路由基线（纯 CPU 离线重放）。

审稿人必问："低置信度时不用 reviewer，直接换多数类/随机标签是否就够？"
设计：对主表 8 种子（hint 配置）的 test_records 重放同一门控协议
（p_a < 0.65 触发，margin=0.05），仅把 critic 提案替换为启发式：
  - neutral-replace：触发桶内无条件改判多数类 neutral（启发式最强口径）
  - random-replace：触发桶内均匀随机改判（同种子可复现）
  - gated（critic）：原门控（对照，应复现主表）
输出 outputs/review_followups/e1_heuristic_routing.json。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation.significance import wf1  # noqa: E402
from debate_erc.utils.label_schema import MELD_LABELS  # noqa: E402

SH = PROJECT_ROOT / "outputs/selective_hetero"
HINT_DIRS = {42: "qwen7b", 43: "qwen7b_s43", 44: "qwen7b_s44",
             **{s: f"qwen7b_s{s}_shared" for s in range(45, 50)}}
TAU, MARGIN = 0.65, 0.05
OUT = PROJECT_ROOT / "outputs/review_followups/e1_heuristic_routing.json"


def replay(rows, policy):
    preds = []
    for r in rows:
        consulted = r["p_a"] is not None and r["p_a"] < TAU
        if policy == "primary":
            preds.append(r["y_a"])
        elif policy == "gated":
            flip = consulted and r["y_b"] not in (None, r["y_a"]) and r["p_b"] > r["p_a"] + MARGIN
            preds.append(r["y_b"] if flip else r["y_a"])
        elif policy == "neutral":
            preds.append("neutral" if consulted else r["y_a"])
        elif policy == "random":
            preds.append("neutral" if consulted else r["y_a"])  # 占位，下面按 idx 重写
        else:
            raise ValueError(policy)
    return preds


def main() -> None:
    per_seed = []
    for s, d in HINT_DIRS.items():
        rows = [json.loads(l) for l in (SH / d / "test_records.jsonl").open()]
        golds = [r["gold"] for r in rows]
        n_trig = sum(1 for r in rows if r["p_a"] is not None and r["p_a"] < TAU)

        rng = np.random.default_rng(4200 + s)
        trig_idx = [i for i, r in enumerate(rows) if r["p_a"] is not None and r["p_a"] < TAU]
        rand_preds = replay(rows, "primary")
        for i in trig_idx:
            rand_preds[i] = str(rng.choice(MELD_LABELS))

        res = {"seed": s, "n": len(rows), "n_triggered": n_trig,
               "trigger_rate": round(n_trig / len(rows), 4)}
        for policy, preds in [
            ("primary", replay(rows, "primary")),
            ("gated", replay(rows, "gated")),
            ("neutral_replace", replay(rows, "neutral")),
            ("random_replace", rand_preds),
        ]:
            res[policy] = round(wf1(preds, golds) * 100, 2)
        per_seed.append(res)

    def agg(key):
        v = [r[key] for r in per_seed]
        return {"mean": round(st.mean(v), 2), "sd": round(st.stdev(v), 2)}

    delta = lambda a, b: [r[a] - r[b] for r in per_seed]
    summary = {}
    for pol in ("gated", "neutral_replace", "random_replace"):
        d = delta(pol, "primary")
        t, p = sps.ttest_rel([r[pol] for r in per_seed], [r["primary"] for r in per_seed])
        summary[pol] = {"delta_vs_primary": {"mean": round(st.mean(d), 2),
                                             "sd": round(st.stdev(d), 2),
                                             "pos": sum(1 for x in d if x > 0),
                                             "paired_t": round(float(t), 3),
                                             "p": round(float(p), 4)},
                        "wf1": agg(pol)}
    summary["primary"] = {"wf1": agg("primary")}

    out = {"protocol": f"frozen gate tau={TAU}, margin={MARGIN}; heuristics replace critic "
                       "proposal unconditionally in the triggered bucket (best case for heuristic)",
           "per_seed": per_seed, "summary": summary}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
