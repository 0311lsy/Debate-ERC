"""Oracle 上界（V3 7.2）：三种路由模式性能上限的离线估计。

给定每个样本在三种模式下的预测标签 {mode: [labels]} 与金标签列表：
- best_mode_assignments：每样本选预测正确的模式（优先轻量 PERCEIVE > COMPOSE > DELIBERATE）；
- upper_bound：按 Oracle 指派计算 Weighted-F1（纯 stdlib 实现，口径同
  sklearn f1_score(average="weighted")）；
- gap：上界 W-F1 − 实际预测的 W-F1（同口径，衡量路由决策还差多少可挖空间）。
"""

from __future__ import annotations

from collections import Counter

from .difficulty_router import RouteMode

# 轻量优先的指派顺序
_MODE_PRIORITY: tuple[RouteMode, ...] = (
    RouteMode.PERCEIVE,
    RouteMode.COMPOSE,
    RouteMode.DELIBERATE,
)


def _weighted_f1(preds: list[str], golds: list[str]) -> float:
    """按类支持度加权的 F1（stdlib 实现，等价 sklearn f1_score average='weighted'）。"""
    if not golds:
        return 0.0
    gold_counts = Counter(golds)
    pred_counts = Counter(preds)
    total = len(golds)
    wf1 = 0.0
    for label, support in gold_counts.items():
        pred_n = pred_counts.get(label, 0)
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        wf1 += (support / total) * f1
    return wf1


class OracleUpperBound:
    """路由 Oracle 上界估计器（离线统计，不依赖模型）。"""

    def __init__(self, predictions: dict, golds: list[str]):
        """
        predictions: {RouteMode 或模式名: 每样本预测标签列表}，三种模式等长；
        golds: 金标签列表。
        """
        self.golds = list(golds)
        self.predictions: dict[RouteMode, list[str]] = {}
        for key, labels in predictions.items():
            mode = RouteMode(key)  # 兼容 RouteMode 成员与模式名字符串
            self.predictions[mode] = list(labels)
        n = len(self.golds)
        for mode, labels in self.predictions.items():
            if len(labels) != n:
                raise ValueError(
                    f"模式 {mode.value} 的预测长度 ({len(labels)}) 与金标签长度 ({n}) 不一致"
                )

    def best_mode_assignments(self) -> list[RouteMode]:
        """每样本选预测正确的模式；多个正确时优先轻量（PERCEIVE > COMPOSE > DELIBERATE）。

        全部模式都错时选最轻量的 PERCEIVE（对上界无影响，仅保证指派确定性）。
        """
        assignments: list[RouteMode] = []
        for i, gold in enumerate(self.golds):
            chosen = _MODE_PRIORITY[0]
            for mode in _MODE_PRIORITY:
                if mode in self.predictions and self.predictions[mode][i] == gold:
                    chosen = mode
                    break
            assignments.append(chosen)
        return assignments

    def upper_bound(self) -> float:
        """Oracle 指派下的 Weighted-F1 上界。"""
        assignments = self.best_mode_assignments()
        preds = [self.predictions[mode][i] for i, mode in enumerate(assignments)]
        return _weighted_f1(preds, self.golds)

    def gap(self, actual_preds: list[str]) -> float:
        """上界 W-F1 − 实际 W-F1（同口径，衡量路由决策还差多少可挖空间）。"""
        if len(actual_preds) != len(self.golds):
            raise ValueError(
                f"actual_preds 长度 ({len(actual_preds)}) 与金标签长度 ({len(self.golds)}) 不一致"
            )
        return self.upper_bound() - _weighted_f1(actual_preds, self.golds)
