"""收敛判定（V3 6.2 步骤 5）。

双方标签一致且置信度均达到阈值 → 辩论收敛。
"""

from __future__ import annotations

from ..data.schemas import AgentOutput


class ConvergenceChecker:
    """辩论收敛判定器。"""

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def is_converged(self, proponent: AgentOutput, critic: AgentOutput) -> bool:
        """双方 label 一致 且 两者 confidence 均 ≥ 阈值。"""
        return (
            proponent.label == critic.label
            and proponent.confidence >= self.confidence_threshold
            and critic.confidence >= self.confidence_threshold
        )
