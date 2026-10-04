#!/usr/bin/env python
"""三分支框架整合：直答 / 异构辩论 / 弃权 的统一 risk-coverage 分析（纯 CPU）。

数据：s42 全量四元组（qwen7b/test_records.jsonl 的 y_a/p_a + cross/
critic42_all_test.jsonl 的全量 y_b/p_b，按 idx 合并，n=2610）。

策略（覆盖集合完全相同，按 p_a 降序，保证配对可比）：
  A 仅弃权（selective-only）：覆盖集作答 y_a；
  DA 弃权+辩论（three-branch）：覆盖集先走门控辩论（τ/margin），再作答；
     - 高置信：直答；中置信：critic 复审翻案；低置信尾部：弃权。
  DAf 变体：按"终判置信度"（翻案取 p_b，否则 p_a）重排覆盖集。

输出：每策略 risk-coverage 曲线、AURC、risk≤5/10% 覆盖率、
弃权 5/10/20/30/40% 的作答集 W-F1/Macro/Acc，及 A-vs-DA 的 bootstrap。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402

OUT = PROJECT_ROOT / "outputs/three_branch"
HET = PROJECT_ROOT / "outputs/selective_hetero"
LABELS = list(MELD_SCHEMA.labels)
TAU, MARGIN = 0.65, 0.05
ABSTAIN_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40]
RISK_BUDGETS = [0.05, 0.10]


def load_quads():
    prop = {}
    for line in (HET / "qwen7b/test_records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        prop[r["idx"]] = r
    crit = {}
    for line in (HET / "cross/critic42_all_test.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        crit[r["idx"]] = r
    rows = []
    for i in sorted(prop):
        pr, cr = prop[i], crit.get(i)
        if cr is None:
            continue
        rows.append({"idx": i, "gold": pr["gold"], "y_a": pr["y_a"], "p_a": pr["p_a"],
                     "y_b": cr["y_b"], "p_b": cr["p_b"]})
    return rows


def debate_prediction(r, tau=TAU, margin=MARGIN):
    """门控辩论终判与终判置信度。"""
    if r["p_a"] < tau and r["y_b"] != r["y_a"] and r["p_b"] > r["p_a"] + margin:
        return r["y_b"], r["p_b"]
    return r["y_a"], r["p_a"]


def risk_coverage(rows, pred_of, conf_key, with_wf1=True):
    """按置信度降序：返回 coverage 序列、risk(=1-acc)、W-F1、AURC。"""
    ordered = sorted(rows, key=lambda r: r[conf_key], reverse=True)
    n = len(ordered)
    covs, risks, wf1s, cum_err = [], [], [], 0
    correct = 0
    for i, r in enumerate(ordered, 1):
        pred = pred_of(r)
        if pred != r["gold"]:
            cum_err += 1
        else:
            correct += 1
        covs.append(i / n)
        risks.append(cum_err / i)
        if with_wf1:
            golds = [x["gold"] for x in ordered[:i]]
            preds = [pred_of(x) for x in ordered[:i]]
            wf1s.append(compute_metrics(preds, golds, LABELS)["wf1"])
    aurc = float(np.trapz(risks, covs))
    return covs, risks, wf1s, aurc, correct / n


def coverage_at_risk(covs, risks, alpha):
    """风险 ≤ α 时的最大覆盖率（取最后一个满足点，与 analyze_abstention 一致）。"""
    best = 0.0
    for c, r in zip(covs, risks):
        if r <= alpha:
            best = c
    return best


def abstain_table(rows, pred_of, conf_key, trials=20, seed=2026):
    """弃权最低置信尾部后，作答集指标；附同比例随机弃权对照。"""
    rng = np.random.default_rng(seed)
    n = len(rows)
    ordered = sorted(rows, key=lambda r: r[conf_key], reverse=True)
    out = {}
    for f in ABSTAIN_FRACTIONS:
        keep = ordered[:max(1, round(n * (1 - f)))]
        m = compute_metrics([pred_of(r) for r in keep], [r["gold"] for r in keep], LABELS)
        # 随机弃权：trials 次平均
        rand_wf1, rand_acc = [], []
        for _ in range(trials):
            idx = rng.choice(n, len(keep), replace=False)
            sub = [rows[i] for i in idx]
            mr = compute_metrics([pred_of(r) for r in sub], [r["gold"] for r in sub], LABELS)
            rand_wf1.append(mr["wf1"])
            rand_acc.append(mr["accuracy"])
        out[f"abstain_{int(f*100)}pct"] = {
            "coverage": round(len(keep) / n, 4),
            "wf1": round(m["wf1"], 4), "macro_f1": round(m["macro_f1"], 4),
            "accuracy": round(m["accuracy"], 4), "risk": round(1 - m["accuracy"], 4),
            "random_wf1": round(float(np.mean(rand_wf1)), 4),
            "random_accuracy": round(float(np.mean(rand_acc)), 4)}
    return out


def bootstrap_aurc_diff(rows, pred_a, pred_d, key, B=2000, seed=0):
    """DA 相对 A 的 AURC 差值 bootstrap（负值=辩论降低风险面积）。"""
    rng = np.random.default_rng(seed)
    n = len(rows)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        sub = [rows[i] for i in idx]
        a_a = risk_coverage(sub, pred_a, key, with_wf1=False)[3]
        a_d = risk_coverage(sub, pred_d, key, with_wf1=False)[3]
        diffs.append(a_d - a_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_quads()
    n = len(rows)
    labels = sorted({r["gold"] for r in rows})
    for r in rows:
        r["y_d"], r["p_final"] = debate_prediction(r)

    policies = {
        "A_abstain_only": (lambda r: r["y_a"], "p_a"),
        "DA_debate_plus_abstain": (lambda r: r["y_d"], "p_a"),
        "DAf_final_confidence": (lambda r: r["y_d"], "p_final"),
    }
    result = {"n": n, "tau": TAU, "margin": MARGIN, "policies": {}}
    curves = {}
    for name, (pred_of, key) in policies.items():
        covs, risks, wf1s, aurc, acc = risk_coverage(rows, pred_of, key)
        curves[name] = {"coverage": covs, "risk": risks, "wf1": wf1s}
        result["policies"][name] = {
            "ordering_conf": key, "full_accuracy": round(acc, 4),
            "full_wf1": round(compute_metrics([pred_of(r) for r in rows],
                                              [r["gold"] for r in rows], LABELS)["wf1"], 4),
            "aurc": round(aurc, 6),
            "coverage_at_risk": {f"risk<={a}": round(coverage_at_risk(covs, risks, a), 4)
                                 for a in RISK_BUDGETS},
            "abstention": abstain_table(rows, pred_of, key)}

    # oracle 曲线（错误样本最先弃权），两种预测各一条
    oracles = {}
    for name, pred_of in [("A", lambda r: r["y_a"]), ("DA", lambda r: r["y_d"])]:
        ordered = sorted(rows, key=lambda r: pred_of(r) != r["gold"])
        covs, risks = [], []
        cum_err = 0
        for i, r in enumerate(ordered, 1):
            if pred_of(r) != r["gold"]:
                cum_err += 1
            covs.append(i / n)
            risks.append(cum_err / i)
        oracles[name] = {"coverage": covs, "risk": risks,
                         "aurc": float(np.trapz(risks, covs))}
    result["oracle_aurc"] = {"A": round(oracles["A"]["aurc"], 6),
                             "DA": round(oracles["DA"]["aurc"], 6)}

    mean_d, lo, hi = bootstrap_aurc_diff(
        rows, lambda r: r["y_a"], lambda r: r["y_d"], "p_a")
    result["aurc_diff_DA_minus_A"] = {"mean": round(mean_d, 6),
                                      "ci95": [round(lo, 6), round(hi, 6)]}

    (OUT / "three_branch_risk_coverage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(OUT / "curves.npz",
             **{f"{k}_{m}": np.array(v[m]) for k, v in curves.items()
                for m in ("coverage", "risk", "wf1")},
             oracle_A_cov=np.array(oracles["A"]["coverage"]),
             oracle_A_risk=np.array(oracles["A"]["risk"]),
             oracle_DA_cov=np.array(oracles["DA"]["coverage"]),
             oracle_DA_risk=np.array(oracles["DA"]["risk"]))

    print(f"n={n}  τ={TAU} margin={MARGIN}")
    print(f"{'策略':<26}{'full W-F1':>10}{'full Acc':>10}{'AURC':>9}"
          f"{'cov@r≤5%':>10}{'cov@r≤10%':>11}")
    for name, e in result["policies"].items():
        print(f"{name:<26}{e['full_wf1']:>10.4f}{e['full_accuracy']:>10.4f}"
              f"{e['aurc']:>9.4f}{e['coverage_at_risk']['risk<=0.05']*100:>9.1f}%"
              f"{e['coverage_at_risk']['risk<=0.1']*100:>10.1f}%")
    print(f"\noracle AURC：仅弃权 {result['oracle_aurc']['A']:.4f} | "
          f"弃权+辩论 {result['oracle_aurc']['DA']:.4f}")
    print(f"AURC 差值（DA−A）：{mean_d:+.5f}，95% CI [{lo:+.5f}, {hi:+.5f}]"
          f"{'（显著为负→辩论在同覆盖下降低风险）' if hi < 0 else ''}")
    print("\n各弃权比例下作答集 W-F1：")
    print(f"{'弃权比例':>8}{'仅弃权':>10}{'弃权+辩论':>12}{'Δ':>8}")
    for f in ABSTAIN_FRACTIONS:
        k = f"abstain_{int(f*100)}pct"
        a = result["policies"]["A_abstain_only"]["abstention"][k]["wf1"]
        da = result["policies"]["DA_debate_plus_abstain"]["abstention"][k]["wf1"]
        print(f"{int(f*100):>7}%{a:>10.4f}{da:>12.4f}{da-a:>+8.4f}")


if __name__ == "__main__":
    main()
