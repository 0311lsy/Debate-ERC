"""提出方 Agent（Agent1）：因果链论证（V3 6.1 / 设计报告 4.2）。

渲染 causal_chain.j2 提示（Trigger → State change → Label → Confidence），
支持辩论轮次：critique_points（批判方分歧点）与 history（前轮摘要）。
"""

from __future__ import annotations

from ..data.schemas import AgentOutput, DialogueContext
from .base_agent import BaseAgent
from .prompt_builder import render


def _text_between(text: str, start_marker: str, end_marker: str) -> str:
    """截取 start_marker 之后、end_marker 之前的文本；任一标记缺失时做合理截断。"""
    s = text.find(start_marker)
    if s == -1:
        return ""
    s += len(start_marker)
    e = text.find(end_marker, s)
    return text[s:e] if e != -1 else text[s:]


class CausalProponentAgent(BaseAgent):
    """Agent1 "提出方"：为目标话语构建因果情感链并给出候选标签。"""

    def infer(
        self,
        context: DialogueContext,
        reasoning_hint: str | None = None,
        critique_points: list[str] | None = None,
        history: str | None = None,
    ) -> AgentOutput:
        """渲染 causal_chain.j2 并生成。

        critique_points: 批判方（Agent2）提出的分歧点，辩论轮次中要求逐条回应；
        history: 前几轮辩论的摘要文本。
        """
        prompt = render(
            "causal_chain.j2",
            dialogue=context.render(),
            target_speaker=context.target_speaker,
            target_utterance=context.target_utterance,
            labels=list(self.schema.labels),
            critique_points=critique_points,
            history=history,
        )
        prompt = self._inject_hint(prompt, reasoning_hint)
        text = self._generate(prompt)

        label, confidence = self._parse_label_and_confidence(text)
        causal_chain = _text_between(text, "Trigger:", "Label:").strip()
        return AgentOutput(
            label=label,
            confidence=confidence,
            causal_chain=causal_chain,
            raw_text=text,
        )

    def infer_reask(
        self,
        context: DialogueContext,
        disagreement_points: list[str],
        previous_label: str,
        previous_confidence: float,
        reasoning_hint: str | None = None,
    ) -> AgentOutput:
        """Re-ask 兜底路径（V3 6.7.3）：渲染 reask.j2 要求 Agent1 重新推导结论。

        disagreement_points: 辩论阶段的分歧点（逐条回应）；
        previous_label / previous_confidence: 上一轮结论（re-derive 的起点，禁止复读）。
        """
        prompt = render(
            "reask.j2",
            dialogue=context.render(),
            target_speaker=context.target_speaker,
            target_utterance=context.target_utterance,
            disagreement_points=disagreement_points,
            previous_label=previous_label,
            previous_confidence=previous_confidence,
            labels=list(self.schema.labels),
        )
        prompt = self._inject_hint(prompt, reasoning_hint)
        text = self._generate(prompt)

        label, confidence = self._parse_label_and_confidence(text)
        causal_chain = _text_between(text, "Re-analysis:", "Label:").strip()
        return AgentOutput(
            label=label,
            confidence=confidence,
            causal_chain=causal_chain,
            raw_text=text,
        )
