"""Class-Balanced 少数类策略（V3 7.3）。

- class_balanced_weights：CB 有效样本数权重（Cui et al., 2019, Class-Balanced Loss），
  w_c = (1-β)/(1-β^{n_c})，稀有类（fear/disgust）获得更高权重，归一化使均值为 1；
- class_balanced_sampler_labels：class-balanced 重采样索引（每类采样数 = 总数/类数，
  多数类随机欠采样、少数类均匀循环采样），供训练前构造均衡子集。

两者分别对应消融开关 fewshot.class_balanced 与 focal_loss 的配套权重来源。
"""

from __future__ import annotations

import random
from collections import Counter
from math import isclose

# 默认随机源（S7 契约）：rng=None 时使用，多次调用共享状态演进
_DEFAULT_SAMPLER_RNG = random.Random(42)


def class_balanced_weights(labels: list[str], beta: float = 0.9999) -> dict[str, float]:
    """Class-Balanced 权重 (1-β)/(1-β^n_c)，归一化使均值为 1。

    参数：
        labels: 训练集标签序列（如 MELD 7 类的每样本 gold 标签）
        beta: CB 温度；β→1 时趋向逆频率权重，β=0 时退化为均匀权重

    返回：
        {label: weight}，权重均值归一为 1（可直接作为 focal/CE 的类别权重，
        均值归一保证与"权重=1"的基线 loss 量级可比）
    """
    if not labels:
        return {}
    counts = Counter(labels)
    raw: dict[str, float] = {}
    for cls, n_c in counts.items():
        n = max(int(n_c), 1)
        denom = 1.0 - (beta ** n)
        if abs(denom) < 1e-12:  # β≈1 且 n 极小的退化情形
            raw[cls] = 1.0
        else:
            raw[cls] = (1.0 - beta) / denom
    mean = sum(raw.values()) / len(raw)
    if isclose(mean, 0.0):
        return {cls: 1.0 for cls in raw}
    return {cls: w / mean for cls, w in raw.items()}


def class_balanced_sampler_labels(
    labels: list[str], rng: random.Random | None = None
) -> list[int]:
    """class-balanced 重采样：返回采样后的索引列表。

    规则：
    - 每类采样数 = 总数 // 类数（类数取 labels 中实际出现的类）
    - 多数类（样本数 > 目标数）：随机欠采样（不放回抽 n_per_class 个）
    - 少数类（样本数 < 目标数）：均匀循环采样（cycle，重复采满目标数）
    - 返回按索引升序排序（shuffle 交给 DataLoader）

    参数：
        labels: 训练集标签序列
        rng: 随机源；None → 模块级 random.Random(42)（注入可保证测试/复现确定性）

    例：labels = [a, a, a, a, b, b] → 每类 3 个：
      a 类（4 个，多数类）随机欠采样 3 个，b 类（2 个，少数类）循环采样 [4, 5, 4]；
      rng=Random(42) 时 a 类抽到 [0, 3, 1]，最终返回 [0, 1, 3, 4, 4, 5]。
    """
    if not labels:
        return []
    sampler = rng if rng is not None else _DEFAULT_SAMPLER_RNG
    by_class: dict[str, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(idx)
    n_total = len(labels)
    n_per_class = max(n_total // len(by_class), 1)
    sampled: list[int] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        if len(pool) > n_per_class:
            picked = sampler.sample(pool, n_per_class)  # 多数类随机欠采样
        else:
            picked = [pool[j % len(pool)] for j in range(n_per_class)]  # 少数类均匀循环
        sampled.extend(picked)
    return sorted(sampled)
