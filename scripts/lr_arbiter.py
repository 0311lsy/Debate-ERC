#!/usr/bin/env python
"""逻辑回归仲裁器离线实验（纯 CPU）：回应"oracle 上界 76.1 vs 规则门控 69.6"的 6 点缺口。

协议（与主实验同口径）：
- 训练：dev 四元组（1108 条），仅在 y_a != y_b 的分歧样本上学"P(y_b 正确 | 特征)"
- 评估：test 四元组（2610 条）只评一次，最终预测 = 分歧且 LR 判 y_b 时取 y_b，否则取 y_a
- 对照：规则门控（tau=0.65/margin=0.05）、always-flip-on-disagree、conf-pick、oracle

特征（避免泄漏 gold）：
  p_a, p_b, p_b-p_a, |p_a-p_b|, low_conf = p_a<0.65,
  y_a 与 y_b 的 dev 先验错误率（dev 上各类被主/复审模型判错的频率）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402

LABELS = list(MELD_SCHEMA.labels)
HET = PROJECT_ROOT / "outputs/selective_hetero/qwen7b"
OUT = PROJECT_ROOT / "outputs/arbiter"
TAU, MARGIN = 0.65, 0.05


def load(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return sorted(rows, key=lambda r: r["idx"])


def load_test_quads():
    """test_records 只保留触发样本的 critic 输出，需与 cross 全量转储按 idx 合并。"""
    prop = {r["idx"]: r for r in load(HET / "test_records.jsonl")}
    crit = {r["idx"]: r for r in load(HET.parent / "cross/critic42_all_test.jsonl")}
    out = []
    for i in sorted(prop):
        if i not in crit:
            continue
        r = dict(prop[i])
        r["y_b"] = crit[i]["y_b"]
        r["p_b"] = crit[i]["p_b"]
        out.append(r)
    return out


def class_error_prior(rows, key_pred):
    """dev 上各类别作为预测值时的错误率（gold 先验不参与特征）。"""
    tot = {c: 0 for c in LABELS}
    err = {c: 0 for c in LABELS}
    for r in rows:
        tot[r[key_pred]] += 1
        if r[key_pred] != r["gold"]:
            err[r[key_pred]] += 1
    return {c: (err[c] / tot[c] if tot[c] else 0.5) for c in LABELS}


def feats(r, prior_a, prior_b):
    pa, pb = r["p_a"], r["p_b"]
    return [pa, pb, pb - pa, abs(pa - pb), float(pa < TAU),
            prior_a[r["y_a"]], prior_b[r["y_b"]]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load(HET / "dev_records.jsonl")
    test = load_test_quads()
    prior_a = class_error_prior(dev, "y_a")
    prior_b = class_error_prior(dev, "y_b")

    # 分歧子集训练
    d_tr = [r for r in dev if r["y_a"] != r["y_b"]]
    X = np.array([feats(r, prior_a, prior_b) for r in d_tr])
    y = np.array([float(r["y_b"] == r["gold"]) for r in d_tr])
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y)

    def predict(rows, mode):
        pred = []
        stats = {"n_flip": 0, "rescue": 0, "harm": 0}
        for r in rows:
            ya, yb = r["y_a"], r["y_b"]
            final = ya
            if ya != yb:
                take_b = False
                if mode == "lr":
                    p = lr.predict_proba(np.array(feats(r, prior_a, prior_b)).reshape(1, -1))[0, 1]
                    take_b = p > 0.5
                elif mode == "rule":
                    take_b = r["p_a"] < TAU and r["p_b"] > r["p_a"] + MARGIN
                elif mode == "always":
                    take_b = True
                elif mode == "conf":
                    take_b = r["p_b"] > r["p_a"]
                if take_b:
                    final = yb
                    stats["n_flip"] += 1
                    if yb == r["gold"] and ya != r["gold"]:
                        stats["rescue"] += 1
                    if ya == r["gold"] and yb != r["gold"]:
                        stats["harm"] += 1
            pred.append(final)
        gold = [r["gold"] for r in rows]
        m = compute_metrics(pred, gold, LABELS)
        return m, stats

    results = {}
    for mode in ("lr", "rule", "always", "conf"):
        m, s = predict(test, mode)
        results[mode] = {"wf1": round(m["wf1"] * 100, 2),
                         "macro_f1": round(m["macro_f1"] * 100, 2),
                         "acc": round(m["accuracy"] * 100, 2), **s}

    # 基线与 oracle
    gold = [r["gold"] for r in test]
    base = compute_metrics([r["y_a"] for r in test], gold, LABELS)
    oracle_pred = [r["y_a"] if r["y_a"] == r["gold"] else r["y_b"] for r in test]
    oracle = compute_metrics(oracle_pred, gold, LABELS)
    results["proponent_only"] = {"wf1": round(base["wf1"] * 100, 2)}
    results["oracle"] = {"wf1": round(oracle["wf1"] * 100, 2)}

    # LR 决策阈值扫描：精度-召回前沿（回应 oracle 缺口）
    sweep = []
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        pred, s = [], {"n_flip": 0, "rescue": 0, "harm": 0}
        for r in test:
            final = r["y_a"]
            if r["y_a"] != r["y_b"]:
                p = lr.predict_proba(np.array(feats(r, prior_a, prior_b)).reshape(1, -1))[0, 1]
                if p > thr:
                    final = r["y_b"]
                    s["n_flip"] += 1
                    if r["y_b"] == r["gold"] and r["y_a"] != r["gold"]:
                        s["rescue"] += 1
                    if r["y_a"] == r["gold"] and r["y_b"] != r["gold"]:
                        s["harm"] += 1
            pred.append(final)
        m = compute_metrics(pred, gold, LABELS)
        prec = s["rescue"] / s["n_flip"] if s["n_flip"] else 0.0
        sweep.append({"thr": thr, "wf1": round(m["wf1"] * 100, 2),
                      "flip_precision": round(prec, 3), **s})
    results["lr_threshold_sweep"] = sweep

    # 分歧池上限：test 分歧样本中 y_b 对/y_a 错的潜在 rescue 总数
    dis = [r for r in test if r["y_a"] != r["y_b"]]
    results["disagreement_pool"] = {
        "n": len(dis),
        "potential_rescue": sum(1 for r in dis if r["y_b"] == r["gold"] and r["y_a"] != r["gold"]),
        "potential_harm": sum(1 for r in dis if r["y_a"] == r["gold"] and r["y_b"] != r["gold"]),
        "both_wrong": sum(1 for r in dis if r["y_a"] != r["gold"] and r["y_b"] != r["gold"]),
        "yb_correct_rate": round(float(np.mean([r["y_b"] == r["gold"] for r in dis])), 4),
    }

    # dev 上的 LR 训练拟合质量（防止过拟合误判）
    p_dev = lr.predict_proba(X)[:, 1]
    dev_auc_proxy = float(np.mean((p_dev > 0.5) == y))
    results["_meta"] = {
        "n_dev": len(dev), "n_test": len(test),
        "n_dev_disagree": len(d_tr),
        "dev_disagree_rate": round(len(d_tr) / len(dev), 4),
        "train_acc_threshold05": round(dev_auc_proxy, 4),
        "lr_coef": [round(float(c), 4) for c in lr.coef_[0]],
        "feature_names": ["p_a", "p_b", "p_b-p_a", "|p_a-p_b|", "low_conf",
                          "prior_err_ya", "prior_err_yb"],
    }

    (OUT / "lr_arbiter.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
