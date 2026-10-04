"""辩论主循环（V3 6.1 / 6.2）。

流程：
1. 第 1 轮：proponent.infer → critic.infer(proposal=Agent1 完整输出)；
2. 每轮结束用 ConvergenceChecker.is_converged 判定；未收敛且未达 max_rounds
   则进入下一轮：proponent.infer(critique_points=critic 分歧点, history=前轮摘要)，
   critic 再次审查；
3. 收敛 → final_label = 共同标签，confidence = 双方均值；
   max_rounds 未收敛 → final_label = proponent.label（converged=False）；
4. ReaskFallback 存在且 should_reask 为真 → outcome.reask_triggered=True 并进入兜底循环；
5. consensus_but_wrong 留 False（由评估阶段对照金标签填充）。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..data.schemas import AgentOutput, DebateOutcome, DialogueContext
from .convergence import ConvergenceChecker
from .reask_fallback import ReaskFallback

if TYPE_CHECKING:
    from ..agents import BaseAgent


class DebateLoop:
    """Agent1 ↔ Agent2 多轮辩论主循环。"""

    def __init__(
        self,
        proponent: BaseAgent,
        critic: BaseAgent,
        convergence: ConvergenceChecker,
        max_rounds: int = 3,
        reask: ReaskFallback | None = None,
    ):
        self.proponent = proponent
        self.critic = critic
        self.convergence = convergence
        self.max_rounds = max_rounds
        self.reask = reask

    def run(self, context: DialogueContext, reasoning_hint: str | None = None) -> DebateOutcome:
        """对单个样本执行完整辩论，返回 DebateOutcome。"""
        start = time.time()

        a1 = self.proponent.infer(context, reasoning_hint=reasoning_hint)
        a2 = self.critic.infer(context, proposal=a1.raw_text, reasoning_hint=reasoning_hint)
        rounds = 1
        generation_calls = 2
        first_round_label = a1.label
        history_lines: list[str] = [self._round_summary(rounds, a1, a2)]
        converged = self.convergence.is_converged(a1, a2)

        while not converged and rounds < self.max_rounds:
            # 轮次注入兜底：critic 未给出分歧点但双方标签不一致时，注入标签分歧
            critique_points = a2.critique_points
            if not critique_points and a1.label != a2.label:
                critique_points = [f"label disagreement: {a1.label} vs {a2.label}"]
            rounds += 1
            a1 = self.proponent.infer(
                context,
                reasoning_hint=reasoning_hint,
                critique_points=critique_points,
                history="\n".join(history_lines),
            )
            a2 = self.critic.infer(context, proposal=a1.raw_text, reasoning_hint=reasoning_hint)
            generation_calls += 2
            history_lines.append(self._round_summary(rounds, a1, a2))
            converged = self.convergence.is_converged(a1, a2)

        if converged:
            final_label = a1.label  # 双方一致
            confidence = (a1.confidence + a2.confidence) / 2.0
        else:
            final_label = a1.label
            confidence = a1.confidence

        disagreement_points = list(a2.critique_points)
        if not disagreement_points and a1.label != a2.label:
            # 分歧点兜底：仅在双方标签确实不一致时生成（收敛时不应出现伪分歧）
            disagreement_points = [f"label disagreement: {a1.label} vs {a2.label}"]

        outcome = DebateOutcome(
            final_label=final_label,
            confidence=confidence,
            rounds=rounds,
            converged=converged,
            proponent_output=a1,
            critic_output=a2,
            disagreement_points=disagreement_points,
            final_reasoning=a1.causal_chain or a1.raw_text,
            first_round_label=first_round_label,
            generation_calls=generation_calls,
        )

        if self.reask is not None and self.reask.should_reask(a1, a2, converged, confidence):
            outcome.reask_triggered = True
            outcome = self.reask.reask_loop(
                context, self.proponent, outcome, reasoning_hint=reasoning_hint
            )

        outcome.latency_seconds = time.time() - start
        return outcome

    @staticmethod
    def _round_summary(round_no: int, a1: AgentOutput, a2: AgentOutput) -> str:
        """前轮摘要（注入下一轮 Agent1 的 history 槽位）。"""
        points = "; ".join(a2.critique_points) if a2.critique_points else "none"
        return (
            f"Round {round_no}: Agent1 -> {a1.label} "
            f"(confidence {a1.confidence:.2f}); "
            f"Agent2 -> {a2.label} (confidence {a2.confidence:.2f}); "
            f"critique: {points}"
        )
