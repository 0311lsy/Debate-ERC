#!/usr/bin/env python
"""3 proponent × 3 critic 因子交叉分析（τ=0.65/margin=0.05 锁定，test 零调参）。

数据：
  proponent i 全样本 (y_a,p_a)：pair i 的 test_records（每行必有 y_a/p_a）
  critic j 全样本 (y_b,p_b)  ：cross/critic{j}_all_test.jsonl
  dev 同理（各自 pair dev_records 本就全样本含 y_b）。
输出：ΔW-F1 的 3×3 矩阵、行列边际均值、9 格总均值±std，
      以及 critic 独立 W-F1——用于归因 s43 失败（critic 弱 / proponent 特殊 / 交互）。
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "src")
from debate_erc.evaluation import compute_metrics
from debate_erc.utils.label_schema import MELD_SCHEMA
L = list(MELD_SCHEMA.labels)
ROOT = Path("outputs/selective_hetero")
TAU, MARGIN = 0.65, 0.05

PAIR_DIR = {"42": "qwen7b", "43": "qwen7b_s43_shared", "44": "qwen7b_s44_shared"}
SEEDS = ["42", "43", "44"]


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def load_split_data(split):
    props, crits = {}, {}
    for s in SEEDS:
        if split == "test":
            rows = read_jsonl(ROOT / PAIR_DIR[s] / "test_records.jsonl")
            props[s] = {r["idx"]: (r["y_a"], r["p_a"]) for r in rows}
            crits[s] = {r["idx"]: (r["y_b"], r["p_b"]) for r in
                        read_jsonl(ROOT / "cross" / f"critic{s}_all_test.jsonl")}
        else:
            rows = read_jsonl(ROOT / PAIR_DIR[s].replace("_shared", "") /
                              f"{split}_records.jsonl")
            # pair i 的 dev 文件里有 prop_i 与 critic_i 的全样本输出
            props[s] = {r["idx"]: (r["y_a"], r["p_a"]) for r in rows}
            crits[s] = {r["idx"]: (r["y_b"], r["p_b"]) for r in rows}
    return props, crits


def paired_delta(prop, crit, idxs, golds):
    ya = [prop[i][0] for i in idxs]
    yb = [crit[i][0] for i in idxs]
    pa = [prop[i][1] for i in idxs]
    pb = [crit[i][1] for i in idxs]
    base = compute_metrics(ya, golds, L)["wf1"]
    final = [b if (x < TAU and a != b and y > x + MARGIN) else a
             for a, b, x, y in zip(ya, yb, pa, pb)]
    m = compute_metrics(final, golds, L)
    return (m["wf1"] - base) * 100, base * 100


def matrix(split, tag):
    props, crits = load_split_data(split)
    idxs = sorted(props["42"].keys())
    golds_src = (read_jsonl(ROOT / PAIR_DIR["42"] / f"{split}_records.jsonl")
                 if split == "test"
                 else read_jsonl(ROOT / "qwen7b" / f"{split}_records.jsonl"))
    # gold 与 idx 对齐
    gold_map = {r["idx"]: r["gold"] for r in golds_src}
    golds = [gold_map[i] for i in idxs]
    print(f"\n=== {tag}：ΔW-F1（百分点）3×3 矩阵，τ={TAU}/margin={MARGIN} ===")
    print("         critic42 critic43 critic44 | 行均(prop)")
    grid = {}
    col = {c: [] for c in SEEDS}
    for ps in SEEDS:
        vals = []
        for cs in SEEDS:
            d, _ = paired_delta(props[ps], crits[cs], idxs, golds)
            grid[(ps, cs)] = d; vals.append(d); col[cs].append(d)
        print(f" prop{ps}  " + " ".join(f"{v:+8.2f}" for v in vals) +
              f" | {st.mean(vals):+6.2f}")
    print("列均(crit) " + " ".join(f"{st.mean(col[c]):+8.2f}" for c in SEEDS))
    allv = list(grid.values())
    diag = [grid[(s, s)] for s in SEEDS]
    off = [v for k, v in grid.items() if k[0] != k[1]]
    print(f"9 格总均值 {st.mean(allv):+.2f} ± {st.stdev(allv):.2f}；"
          f"对角(同seed) {st.mean(diag):+.2f}；非对角 {st.mean(off):+.2f} ± {st.stdev(off):.2f}")
    print(f"9 格中正比例：{sum(v>0 for v in allv)}/9")
    return grid


g_test = matrix("test", "TEST")
g_dev = matrix("dev", "DEV")

# critic 独立能力（全 test）
print("\n=== critic 独立 W-F1（全 test）===")
for s in SEEDS:
    rows = read_jsonl(ROOT / "cross" / f"critic{s}_all_test.jsonl")
    w = compute_metrics([r["y_b"] for r in rows], [r["gold"] for r in rows], L)["wf1"]
    print(f"critic{s}: {w*100:.2f}")
