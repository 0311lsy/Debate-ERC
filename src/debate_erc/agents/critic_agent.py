"""批判方 Agent（Agent2）：三类缺陷审查（V3 6.1 / 设计报告 4.2）。

渲染 critique.j2 提示，对 Agent1 的因果链提案做三类缺陷审查：
1. 触发-情感不匹配；2. 忽视上下文转折；3. 字面-语境矛盾（反讽）。
同时给出批判方自己的标签与置信度。
"""

from __future__ import annotations

from ..data.schemas import AgentOutput, DialogueContext
from .base_agent import BaseAgent
from .prompt_builder import render


def _parse_critique_points(text: str) -> list[str]:
    """解析 "Critique:" 之后以 "- " 开头的各行（过滤 "none"）。

    缺 "Critique:" 锚时输出格式不符（视为未按模板作答），返回空列表，
    不再从全文开头误采 "- " 行。
    """
    window = text or ""
    marker = "Critique:"
    idx = window.find(marker)
    if idx == -1:
        return []
    window = window[idx + len(marker):]
    points: list[str] = []
    for line in window.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            point = stripped[2:].strip()
            if point and point.lower() != "none":
                points.append(point)
    return points


class CriticAgent(BaseAgent):
    """Agent2 "批判方"：审查 Agent1 提案并给出独立判断。"""

    def infer(
        self,
        context: DialogueContext,
        reasoning_hint: str | None = None,
        proposal: str = "",
    ) -> AgentOutput:
        """渲染 critique.j2 并生成。

        proposal: Agent1 的完整输出文本（因果链提案）。
        """
        prompt = render(
            "critique.j2",
            dialogue=context.render(),
            target_speaker=context.target_speaker,
            target_utterance=context.target_utterance,
            labels=list(self.schema.labels),
            proposal=proposal,
        )
        prompt = self._inject_hint(prompt, reasoning_hint)
        text = self._generate(prompt)

        label, confidence = self._parse_label_and_confidence(text)
        critique_points = _parse_critique_points(text)
        return AgentOutput(
            label=label,
            confidence=confidence,
            critique_points=critique_points,
            raw_text=text,
        )
