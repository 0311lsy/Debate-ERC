#!/usr/bin/env python
"""纯 CPU 离线分析（不加载模型，不占 GPU）：
1) 朴素集成基线 vs 选择性翻案：在完全相同的两个模型输出上比较
   prop / critic / confidence-pick / 无门控按margin翻 / ours(τ=0.65,m=0.05) / oracle，
   同时报告 critic 调用率（推理成本）；
2) τ 敏感性（dev 与 test 同图口径）与成本-增益；
3) 逐情绪类别 ΔF1 与翻案方向归因。

数据源（对角同 seed 配对，3 seeds）：
  proponent 全样本 (y_a,p_a)：pair 目录 test_records.jsonl（每行必有）
  critic   全样本 (y_b,p_b) ：cross/critic{i}_all_test.jsonl（全 2610 条）
  dev：各 pair dev_records.jsonl（dev 收集了全部样本 y_b）。
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, "src")
from sklearn.metrics import f1_score, accuracy_score
from debate_erc.utils.label_schema import MELD_SCHEMA

L = list(MELD_SCHEMA.labels)
ROOT = Path("outputs/selective_hetero")
TAU, MARGIN = 0.65, 0.05
SEED_DIRS = {"42": "qwen7b", "43": "qwen7b_s43_shared", "44": "qwen7b_s44_shared"}
DEV_DIRS = {"42": "qwen7b", "43": "qwen7b_s43", "44": "qwen7b_s44"}


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def load_test(s):
    prop_rows = read_jsonl(ROOT / SEED_DIRS[s] / "test_records.jsonl")
    crit_rows = read_jsonl(ROOT / "cross" / f"critic{s}_all_test.jsonl")
    cmap = {r["idx"]: (r["y_b"], r["p_b"]) for r in crit_rows}
    out = []
    for r in prop_rows:
        yb, pb = cmap[r["idx"]]
        out.append({"idx": r["idx"], "gold": r["gold"],
                    "ya": r["y_a"], "pa": r["p_a"], "yb": yb, "pb": pb})
    return out


def load_dev(s):
    rows = read_jsonl(ROOT / DEV_DIRS[s] / "dev_records.jsonl")
    return [{"idx": r["idx"], "gold": r["gold"], "ya": r["y_a"], "pa": r["p_a"],
             "yb": r["y_b"], "pb": r["p_b"]} for r in rows]


def m_pred(r, policy):
    """返回 (预测, 是否调用critic)。"""
    if policy == "prop":
        return r["ya"], 0
    if policy == "critic":
        return r["yb"], 1
    if policy == "conf_pick":  # 谁置信高听谁——必须每条都问 critic
        return (r["yb"] if r["pb"] > r["pa"] else r["ya"]), 1
    if policy == "flip_ungated":  # 无门控、按 margin 翻——仍需每条问 critic
        if r["yb"] != r["ya"] and r["pb"] > r["pa"] + MARGIN:
            return r["yb"], 1
        return r["ya"], 1
    if policy == "ours":  # 仅 p_a<τ 才调用并可能翻案
        if r["pa"] < TAU and r["yb"] != r["ya"] and r["pb"] > r["pa"] + MARGIN:
            return r["yb"], 1
        return r["ya"], int(r["pa"] < TAU)
    if policy == "oracle":  # 两答案有对的就取对的（上界，需全量critic）
        if r["ya"] == r["gold"]:
            return r["ya"], 1
        return r["yb"], 1
    raise ValueError(policy)


def metrics(rows, policy):
    preds, calls = [], 0
    for r in rows:
        y, c = m_pred(r, policy)
        preds.append(y); calls += c
    golds = [r["gold"] for r in rows]
    return {"wf1": f1_score(golds, preds, labels=L, average="weighted") * 100,
            "macro": f1_score(golds, preds, labels=L, average="macro") * 100,
            "acc": accuracy_score(golds, preds) * 100,
            "call": calls / len(rows) * 100}


POLICIES = ["prop", "critic", "conf_pick", "flip_ungated", "ours", "oracle"]
CN = {"prop": "Proponent only", "critic": "Critic only",
      "conf_pick": "Confidence-pick", "flip_ungated": "无门控按margin翻",
      "ours": f"Ours(τ={TAU})", "oracle": "Oracle上界"}

test = {s: load_test(s) for s in SEED_DIRS}
dev = {s: load_dev(s) for s in DEV_DIRS}

print("=" * 82)
print("TEST：相同两模型输出上的策略对比（3 seeds 均值±std）")
print("=" * 82)
print(f"{'策略':22}{'W-F1':>14}{'Macro':>12}{'Acc':>10}{'critic调用率':>12}")
summary = {}
for p in POLICIES:
    vals = {k: [metrics(test[s], p)[k] for s in test] for k in ("wf1", "macro", "acc", "call")}
    summary[p] = {k: [round(v, 3) for v in vals[k]] for k in vals}
    print(f"{CN[p]:22}{st.mean(vals['wf1']):>9.2f}±{st.stdev(vals['wf1']):.2f}"
          f"{st.mean(vals['macro']):>8.2f}±{st.stdev(vals['macro']):.2f}"
          f"{st.mean(vals['acc']):>8.2f}{st.mean(vals['call']):>10.1f}%")
# 各 seed Ours 明细
print("\n各 seed Ours vs prop / conf_pick：")
for s in test:
    b, c, o = metrics(test[s], "prop"), metrics(test[s], "conf_pick"), metrics(test[s], "ours")
    print(f"  s{s}: prop {b['wf1']:.2f} | conf_pick {c['wf1']:.2f}（调用100%）"
          f" | ours {o['wf1']:.2f}（调用{o['call']:.1f}%） Δo {o['wf1']-b['wf1']:+.2f}")

# ---------------- τ 敏感性 + 成本-增益 ----------------
print("\n" + "=" * 82)
print("τ 敏感性（margin=0.05）：dev/test ΔW-F1 与 critic 调用率")
print("=" * 82)
print(f"{'τ':>5}{'dev ΔW-F1':>12}{'test ΔW-F1':>13}{'test调用率':>11}")


def gated(r, tau):
    if r["pa"] < tau and r["yb"] != r["ya"] and r["pb"] > r["pa"] + MARGIN:
        return r["yb"]
    return r["ya"]


def wf1_of(rows, fn):
    return f1_score([r["gold"] for r in rows], [fn(r) for r in rows],
                    labels=L, average="weighted") * 100


tau_rows = []
for tau in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]:
    dd, tt, cc = [], [], []
    for s in test:
        db = wf1_of(dev[s], lambda r: r["ya"]); tb = wf1_of(test[s], lambda r: r["ya"])
        dd.append(wf1_of(dev[s], lambda r, t=tau: gated(r, t)) - db)
        tt.append(wf1_of(test[s], lambda r, t=tau: gated(r, t)) - tb)
        cc.append(sum(r["pa"] < tau for r in test[s]) / len(test[s]) * 100)
    tau_rows.append({"tau": tau, "dev_delta": st.mean(dd), "test_delta": st.mean(tt),
                     "call": st.mean(cc)})
    print(f"{tau:5.2f}{st.mean(dd):>+11.2f}{st.mean(tt):>+12.2f}{st.mean(cc):>10.1f}%")

# ---------------- 逐类 ΔF1（Ours，3-seed 均值）-----------------
print("\n" + "=" * 82)
print("逐情绪类别 F1：prop → ours（3 seeds 均值，test）")
print("=" * 82)
per_cls = {lab: {"prop": [], "ours": []} for lab in L}
for s in test:
    g = [r["gold"] for r in test[s]]
    yp = [r["ya"] for r in test[s]]
    yo = [m_pred(r, "ours")[0] for r in test[s]]
    fp = f1_score(g, yp, labels=L, average=None, zero_division=0) * 100
    fo = f1_score(g, yo, labels=L, average=None, zero_division=0) * 100
    for i, lab in enumerate(L):
        per_cls[lab]["prop"].append(fp[i]); per_cls[lab]["ours"].append(fo[i])
print(f"{'类别':10}{'prop F1':>10}{'ours F1':>10}{'Δ':>8}")
for lab in L:
    a, b = st.mean(per_cls[lab]["prop"]), st.mean(per_cls[lab]["ours"])
    print(f"{lab:10}{a:>10.2f}{b:>10.2f}{b-a:>+8.2f}")

# ---------------- 翻案方向归因（gold 类别 × 是否救回）-----------------
print("\n翻案归因（ours，3 seeds 合计）：rescue=错→对 / harm=对→错 / neutral=错→错")
tot = {"rescue": 0, "harm": 0, "neutral": 0}
by_gold = {}
for s in test:
    for r in test[s]:
        y, _ = m_pred(r, "ours")
        if y != r["ya"]:
            if r["ya"] != r["gold"] and y == r["gold"]:
                k = "rescue"
            elif r["ya"] == r["gold"] and y != r["gold"]:
                k = "harm"
            else:
                k = "neutral"
            tot[k] += 1
            by_gold.setdefault(r["gold"], {"rescue": 0, "harm": 0, "neutral": 0})[k] += 1
for lab in L:
    if lab in by_gold:
        d = by_gold[lab]
        print(f"  gold={lab:9s} rescue{d['rescue']:3d} harm{d['harm']:3d} neutral{d['neutral']:3d}")
print(f"  合计：rescue{tot['rescue']} harm{tot['harm']} neutral{tot['neutral']}"
      f"（净 {tot['rescue']-tot['harm']:+d}，rescue/harm={tot['rescue']/max(tot['harm'],1):.2f}）")

out = {"policies": {p: {k: {"mean": round(st.mean(v), 3),
                           "std": round(st.stdev(v), 3)}
                        for k, v in summary[p].items()} for p in POLICIES},
       "tau_curve": tau_rows,
       "per_class": {lab: {"prop": round(st.mean(per_cls[lab]["prop"]), 2),
                           "ours": round(st.mean(per_cls[lab]["ours"]), 2)}
                     for lab in L}}
Path("outputs/selective_hetero/cross/ensemble_baselines.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已保存 outputs/selective_hetero/cross/ensemble_baselines.json")
