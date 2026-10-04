"""统计显著性检验（V3 4.4）：bootstrap CI / paired t-test / Cohen's d。

用于多 seed 与系统间对比（如 debate vs single、ablation 之间），
口径：δ = wf1(A) − wf1(B)，bootstrap 重采样得到 95% CI。
"""

from __future__ import annotations

import random

import numpy as np

from .metrics import compute_metrics


def wf1(preds: list[str], golds: list[str]) -> float:
    """默认指标函数：weighted F1（标签集合取 golds ∪ preds 的观察值）。"""
    labels = tuple(sorted(set(golds) | set(preds)))
    if not labels:
        return 0.0
    return compute_metrics(preds, golds, labels)["wf1"]


def bootstrap_ci(
    preds_a: list[str],
    preds_b: list[str],
    golds: list[str],
    metric_fn=wf1,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """配对 bootstrap：返回 (Δ均值, 95% CI 下界, 95% CI 上界)。

    对样本索引有放回重采样 n 次，每次计算 Δ = metric(A) − metric(B)；
    CI 取重采样分布的 2.5% / 97.5% 分位（线性插值，同 numpy.percentile 默认法）。
    """
    if not (len(preds_a) == len(preds_b) == len(golds)):
        raise ValueError(
            f"长度不一致: preds_a={len(preds_a)} preds_b={len(preds_b)} golds={len(golds)}"
        )
    size = len(golds)
    if size == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        res_a = [preds_a[i] for i in idx]
        res_b = [preds_b[i] for i in idx]
        res_g = [golds[i] for i in idx]
        deltas.append(metric_fn(res_a, res_g) - metric_fn(res_b, res_g))
    mean = sum(deltas) / len(deltas)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return mean, lo, hi


def paired_ttest(scores_a: list[float], scores_b: list[float]) -> tuple[float, float]:
    """配对 t 检验（scipy.stats.ttest_rel）：返回 (t 值, p 值)。

    scipy 不可用时抛出带安装提示的 RuntimeError。
    差值零方差时 t 检验无定义：常数非零差 → (±inf, 0.0)（系统间恒定差异，
    符号随差值方向，恒劣为 -inf，R2-m12 修复；p 值退化为 0）；
    全零差 → (0.0, 1.0)（两系统完全一致）。
    """
    try:
        from scipy import stats as _sp_stats
    except ImportError as exc:
        raise RuntimeError(
            "scipy 不可用，无法执行 paired t-test（请安装: pip install scipy）"
        ) from exc
    if len(scores_a) != len(scores_b):
        raise ValueError(f"scores_a 与 scores_b 长度不一致: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) < 2:
        raise ValueError("配对样本数 < 2，无法做 t 检验")
    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    if all(d == diffs[0] for d in diffs):
        if diffs[0] == 0.0:
            return 0.0, 1.0
        return float("inf") if diffs[0] > 0 else float("-inf"), 0.0
    t_stat, p_value = _sp_stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """配对 Cohen's d（d_z）：mean(diffs) / sd(diffs)。

    配对口径效应量（compare_systems 以逐样本 0/1 正确性得分配对）；
    样本不足、长度不一致或差值零方差（sd=0）时返回 0.0。
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"scores_a 与 scores_b 长度不一致: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) < 2:
        return 0.0
    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / len(diffs)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1)
    sd_d = var_d ** 0.5
    if sd_d < 1e-12:  # 防浮点噪声：近似恒定差按 0 处理，避免返回 ~1e14 量级的伪效应量
        return 0.0
    return mean_d / sd_d


def compare_systems(
    preds_a: list[str],
    preds_b: list[str],
    golds: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """汇总两个系统的显著性对比。

    t 检验口径：以逐样本 0/1 正确性得分配对（二值近似；严格应使用 McNemar，
    此处以 p 值作参考，显著性判定以 bootstrap CI 是否跨 0 为主）。
    Cohen's d 口径：配对 d_z = mean(diffs) / sd(diffs)。

    返回：{"delta_wf1", "ci": (lo, hi), "p_value", "cohens_d", "significant"}
        significant = p < 0.05 且 95% CI 不含 0。
    """
    if not (len(preds_a) == len(preds_b) == len(golds)):
        raise ValueError(
            f"长度不一致: preds_a={len(preds_a)} preds_b={len(preds_b)} golds={len(golds)}"
        )
    delta = wf1(preds_a, golds) - wf1(preds_b, golds)
    _mean, lo, hi = bootstrap_ci(
        preds_a, preds_b, golds, n=n_bootstrap, seed=seed
    )
    scores_a = [1.0 if p == g else 0.0 for p, g in zip(preds_a, golds)]
    scores_b = [1.0 if p == g else 0.0 for p, g in zip(preds_b, golds)]
    _t, p_value = paired_ttest(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)
    significant = bool(p_value < 0.05 and (lo > 0 or hi < 0))
    return {
        "delta_wf1": delta,
        "ci": (lo, hi),
        "p_value": p_value,
        "cohens_d": d,
        "significant": significant,
    }
