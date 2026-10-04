#!/usr/bin/env python
"""8-seed 统一口径终审聚合（防标定口径混用）。

所有 seed 一律在固定 τ=0.65 / margin=0.05 下、从同一 test 集的逐条
四元组 (gold, y_a, p_a, y_b, p_b) 离线重算，分母按 idx 对齐一致。
- s42/43/44：proponent 输出（各自终审 records，含全部 y_a/p_a）
  与 cross/critic{42,43,44}_all_test.jsonl（全量 y_b/p_b）按 idx 合并；
- s45-49：qwen7b_s{45..49}_shared/test_records.jsonl（采集门即 0.65）。

输出：8-seed 配对 ΔW-F1、t/bootstrap/Wilcoxon/符号检验、Cohen's d_z、
逐 seed 表 JSON + TXT。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "outputs/selective_hetero/cross"
HET = ROOT / "outputs/selective_hetero"
LABELS = ["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"]
TAU, MARGIN = 0.65, 0.05


def wf1(pred, gold):
    from collections import Counter
    f1s, w = {}, Counter(gold)
    for L in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == L and p == L)
        fp = sum(1 for g, p in zip(gold, pred) if g != L and p == L)
        fn = sum(1 for g, p in zip(gold, pred) if g == L and p != L)
        f1s[L] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return sum(f1s[L] * w[L] for L in LABELS) / len(gold)


def load_idx_map(path, keys):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["idx"]] = {k: r.get(k) for k in keys}
    return out


def build_records(seed):
    """返回 [(gold,y_a,p_a,y_b,p_b), ...]，所有字段非 None。"""
    if seed in (45, 46, 47, 48, 49):
        m = load_idx_map(HET / f"qwen7b_s{seed}_shared/test_records.jsonl",
                         ["gold", "y_a", "p_a", "y_b", "p_b"])
    else:
        tag = {42: "qwen7b", 43: "qwen7b_s43", 44: "qwen7b_s44"}[seed]
        prop = load_idx_map(HET / tag / "test_records.jsonl",
                            ["gold", "y_a", "p_a"])
        crit = load_idx_map(CROSS / f"critic{seed}_all_test.jsonl", ["y_b", "p_b"])
        m = {}
        for i, pr in prop.items():
            if i in crit:
                m[i] = {**pr, **crit[i]}
    recs = []
    for i in sorted(m):
        r = m[i]
        # y_b/p_b 允许为 None（门控未触发即未调用 critic），评估时视为不可翻案
        if all(r.get(k) is not None for k in ("gold", "y_a", "p_a")):
            recs.append((r["gold"], r["y_a"], r["p_a"], r.get("y_b"), r.get("p_b")))
    return recs


def evaluate(recs, tau, margin):
    gold = [r[0] for r in recs]
    base = [r[1] for r in recs]
    final, rc, rw, consult, flip = [], 0, 0, 0, 0
    for g, ya, pa, yb, pb in recs:
        y = ya
        if pa < tau and yb is not None:
            consult += 1  # critic 被实际调用（门控触发）
        if yb is not None and pb is not None and pa < tau and yb != ya and pb > pa + margin:
            y = yb
            flip += 1
            if yb == g and ya != g:
                rc += 1
            elif yb != g and ya == g:
                rw += 1
        final.append(y)
    return (wf1(base, gold), wf1(final, gold),
            consult / len(recs), flip / len(recs), rc, rw)


def main():
    rows = []
    for s in [42, 43, 44, 45, 46, 47, 48, 49]:
        recs = build_records(s)
        b, d, consult, flip, rc, rw = evaluate(recs, TAU, MARGIN)
        rows.append({"seed": s, "n": len(recs), "base": b, "debate": d,
                     "delta": d - b, "consult": consult, "flip": flip,
                     "rc": rc, "rw": rw})
    d = np.array([r["delta"] for r in rows])
    n = len(d)
    t, p_t = stats.ttest_1samp(d, 0)
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(100000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    wil = stats.wilcoxon(d)
    npos = int((d > 0).sum())
    dz = t / np.sqrt(n)  # 配对标准化效应量
    t2, p2 = stats.ttest_1samp(d[1:], 0)

    print(f"统一口径 τ={TAU} margin={MARGIN}（s42/43/44 已用全量 critic 转储在 0.65 重算）")
    print(f"{'seed':>5}{'n':>6}{'base':>8}{'debate':>9}{'Δ(pp)':>8}"
          f"{'调用率':>7}{'翻案率':>7}{'对':>4}{'错':>4}")
    for r in rows:
        print(f"{r['seed']:>5}{r['n']:>6}{r['base']*100:>8.2f}{r['debate']*100:>9.2f}"
              f"{r['delta']*100:>+7.2f}{r['consult']*100:>6.1f}%{r['flip']*100:>6.1f}%"
              f"{r['rc']:>4}{r['rw']:>4}")
    print(f"\nmean Δ = {d.mean()*100:+.3f} pp   std = {d.std(ddof=1)*100:.3f} pp"
          f"   正/负 = {npos}/{n-npos}")
    print(f"paired t = {t:.3f}  p = {p_t:.4f}")
    print(f"Cohen d_z = {dz:.2f}")
    print(f"bootstrap 95% CI = [{lo*100:+.3f}, {hi*100:+.3f}] pp")
    print(f"Wilcoxon p = {wil.pvalue:.4f}")
    print(f"符号检验 p = {stats.binomtest(npos, n, .5).pvalue:.4f}")
    print(f"剔除 s42：mean={d[1:].mean()*100:+.3f}  t={t2:.3f} p={p2:.4f}")

    out = {"tau": TAU, "margin": MARGIN, "rows": rows,
           "mean_delta": float(d.mean()), "std_delta": float(d.std(ddof=1)),
           "t": float(t), "p_t": float(p_t), "cohen_dz": float(dz),
           "ci95": [float(lo), float(hi)], "wilcoxon_p": float(wil.pvalue),
           "sign_p": float(stats.binomtest(npos, n, .5).pvalue),
           "drop_s42": {"mean": float(d[1:].mean()), "t": float(t2), "p": float(p2)}}
    (HET / "aggregate_8seeds_uniform.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {HET/'aggregate_8seeds_uniform.json'}")


if __name__ == "__main__":
    main()
