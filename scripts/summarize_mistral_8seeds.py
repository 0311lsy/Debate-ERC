#!/usr/bin/env python
"""Mistral critic 多种子汇总（纯 CPU，只读 test_metrics.json）。

逐种子读取 outputs/selective_hetero/mistral7b_s{42..49}/test_metrics.json，
输出：逐种子 ΔW-F1 / 触发率 / rescue / harm / proponent / gated，
以及 n 种子的 mean±sd、paired t、Wilcoxon、exact sign test。
落盘 outputs/review_followups/mistral_multiseed_summary.json。
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_followups/mistral_multiseed_summary.json"


def paired_t(deltas: list[float]) -> tuple[float, float]:
    """双侧 paired t（H0: mean=0），返回 (t, p)，df=n-1。"""
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(deltas) / n
    var = sum((x - m) ** 2 for x in deltas) / (n - 1)
    se = math.sqrt(var / n)
    t = m / se
    return t, t_sf(t, n - 1)


def t_sf(t: float, df: int) -> float:
    """Student-t 双侧尾概率（正则化不完全 Beta 函数，纯标准库）。"""
    x = df / (df + t * t)
    # P(|T|>t) = I_{df/(df+t^2)}(df/2, 1/2)
    return betainc_reg(df / 2.0, 0.5, x)


def betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz 连分式
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-10:
            break
    return front * h


def wilcoxon_signed_rank(deltas: list[float]) -> float:
    """双侧 Wilcoxon signed-rank 精确 p（n<=30 枚举符号和）。"""
    nz = sorted((abs(x), 1 if x > 0 else -1) for x in deltas if x != 0)
    n = len(nz)
    # 平均秩（结）
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and nz[j + 1][0] == nz[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    wplus = sum(r for r, (_, sgn) in zip(ranks, nz) if sgn > 0)
    total = sum(ranks)
    wstat = min(wplus, total - wplus)
    # 枚举 2^n 个符号组合（平均秩结时精确枚举仍成立：每秩独立选边）
    from itertools import combinations
    rlist = ranks
    count_ge = 0
    for k in range(n + 1):
        for comb in combinations(range(n), k):
            w = sum(rlist[i] for i in comb)
            if min(w, total - w) <= wstat + 1e-9:
                count_ge += 1
    return count_ge / (2 ** n)


def exact_sign(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return float("nan")
    from math import comb
    k = min(pos, neg)
    one_side = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * one_side)


def main():
    rows = []
    for s in range(42, 50):
        p = ROOT / f"outputs/selective_hetero/mistral7b_s{s}/test_metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        base = d["proponent_only"]["wf1"] * 100
        gated = d["hetero_deliberation"]["wf1"] * 100
        rows.append({"seed": s, "base_wf1": round(base, 3), "gated_wf1": round(gated, 3),
                     "delta": round(gated - base, 3),
                     "trigger_pct": round(d["trigger_rate"] * 100, 2),
                     "rescue": d["flip_correct"], "harm": d["flip_wrong"],
                     "n_flip": d["n_flip"]})
    n = len(rows)
    deltas = [r["delta"] for r in rows]
    pos = sum(x > 0 for x in deltas)
    neg = sum(x < 0 for x in deltas)
    mean = sum(deltas) / n
    sd = (sum((x - mean) ** 2 for x in deltas) / (n - 1)) ** 0.5 if n > 1 else 0.0
    t, tp = paired_t(deltas)
    wp = wilcoxon_signed_rank(deltas) if n >= 1 else float("nan")
    sp = exact_sign(pos, neg)
    R = sum(r["rescue"] for r in rows)
    H = sum(r["harm"] for r in rows)
    trig = sum(r["trigger_pct"] for r in rows) / n
    base_m = sum(r["base_wf1"] for r in rows) / n
    gated_m = sum(r["gated_wf1"] for r in rows) / n
    summary = {"n_seeds": n, "seeds": [r["seed"] for r in rows],
               "per_seed": rows,
               "base_wf1_mean": round(base_m, 3), "gated_wf1_mean": round(gated_m, 3),
               "delta_mean": round(mean, 3), "delta_sd": round(sd, 3),
               "n_positive": pos, "n_negative": neg,
               "paired_t": round(t, 3), "paired_t_p": round(tp, 4),
               "wilcoxon_exact_p": round(wp, 4),
               "sign_exact_two_sided_p": round(sp, 4),
               "trigger_pct_mean": round(trig, 2),
               "total_rescue": R, "total_harm": H,
               "rescue_harm_ratio": round(R / max(H, 1), 3)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"=== Mistral critic {n} 种子（s{rows[0]['seed']}–s{rows[-1]['seed']}）===")
    for r in rows:
        print(f"  s{r['seed']}: {r['base_wf1']:.2f} -> {r['gated_wf1']:.2f}  "
              f"Δ={r['delta']:+.2f}  触发={r['trigger_pct']:.1f}%  R/H={r['rescue']}/{r['harm']}")
    print(f"  mean ΔW-F1 = {mean:+.3f} ± {sd:.3f}   ({pos}/{n} 正)")
    print(f"  paired t = {t:.3f}, p = {tp:.4f}")
    print(f"  Wilcoxon exact p = {wp:.4f} ; sign exact p = {sp:.4f}")
    print(f"  触发率均值 = {trig:.1f}%  合计 R/H = {R}/{H} ({R/max(H,1):.2f}:1)")
    print(f"已保存 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
