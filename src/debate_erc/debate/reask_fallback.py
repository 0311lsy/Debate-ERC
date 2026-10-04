"""Re-ask 兜底（V3 6.7.3 伪代码）：分歧点反馈给 Agent1 重新论证。

设计要点（严格对齐 V3 6.7.3）：
- 重提示（re-argue），非投票：每轮把分歧点作为 critique_points 反馈给 Agent1 重新推导；
- 连续两次 label 一致则提前终止（首轮与辩论阶段 Agent1 的最终结论比较）；
- 上限 MAX_REASK 次强制终止，取最后一次论证结论。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data.schemas import AgentOutput, DebateOutcome, DialogueContext

if TYPE_CHECKING:
    from ..agents import BaseAgent

MAX_REASK = 2


class ReaskFallback:
    """低置信 / 未收敛时的 Re-ask 兜底循环（conclude 路径）。"""

    def __init__(self, low_confidence: float = 0.6, max_reask: int = MAX_REASK):
        self.low_confidence = low_confidence
        self.max_reask = max_reask

    def should_reask(
        self,
        proponent: AgentOutput,
        critic: AgentOutput,
        converged: bool,
        confidence: float,
    ) -> bool:
        """触发条件（V3 6.7.3）：
        条件1：辩论未收敛（label 不一致）；
        条件2：label 一致但 confidence < low_confidence（默认 0.6）。
        """
        if not converged:
            return True
        return confidence < self.low_confidence

    def reask_loop(
        self,
        context: DialogueContext,
        proponent_agent: BaseAgent,
        outcome: DebateOutcome,
        reasoning_hint: str | None = None,
    ) -> DebateOutcome:
        """Re-ask 主循环：就地更新并返回 outcome。

        优先调用 proponent_agent.infer_reask(context, disagreement_points,
        previous_label, previous_confidence, reasoning_hint)（re-derive 专用提示，
        hasattr 探测，Agent 未实现时回退 infer + critique_points）；
        每轮更新 final_label / confidence / final_reasoning / reask_rounds /
        generation_calls（每次生成调用 +1）；
        首次更新前写入 outcome.reask_pre_label（reask 前的 final_label）；
        连续两次 label 一致（与上一轮结论，首轮与辩论阶段结论比较）则提前终止。
        """
        prev_label = (
            outcome.proponent_output.label
            if outcome.proponent_output is not None
            else outcome.final_label
        )
        outcome.reask_pre_label = outcome.final_label
        for _ in range(self.max_reask):
            if hasattr(proponent_agent, "infer_reask"):
                output = proponent_agent.infer_reask(
                    context,
                    outcome.disagreement_points,
                    prev_label,
                    outcome.confidence,
                    reasoning_hint=reasoning_hint,
                )
            else:
                output = proponent_agent.infer(
                    context,
                    reasoning_hint=reasoning_hint,
                    critique_points=outcome.disagreement_points,
                )
            outcome.generation_calls += 1
            outcome.reask_rounds += 1
            outcome.final_label = output.label
            outcome.confidence = output.confidence
            outcome.final_reasoning = output.causal_chain or output.raw_text
            if output.label == prev_label:
                break
            prev_label = output.label
        return outcome
