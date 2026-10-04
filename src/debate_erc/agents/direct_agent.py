"""单 Agent 直接推理（基线路径）：classify.j2 + 首行解析。

- base_sft / no_debate 基线与 SFT 训练目标同源（build_classify_prompt），
  保证"训练-推理一致"（V3 6.2）；
- 路由的轻量模式（PERCEIVE / COMPOSE）也走本路径，注入 reasoning_hint 即 CoT 基线。
"""

from __future__ import annotations

from ..data.schemas import AgentOutput, DialogueContext
from .base_agent import BaseAgent
from .prompt_builder import build_classify_prompt


class DirectClassifyAgent(BaseAgent):
    """单 Agent 直接分类：渲染 classify.j2 生成后按正则/首行解析标签与置信度。"""

    def infer(self, context: DialogueContext, reasoning_hint: str | None = None) -> AgentOutput:
        prompt = build_classify_prompt(
            context, self.schema, reasoning_hint=reasoning_hint
        )
        text = self._generate(prompt)
        label, confidence = self._parse_label_and_confidence(text)
        return AgentOutput(
            label=label,
            confidence=confidence,
            raw_text=text,
        )
