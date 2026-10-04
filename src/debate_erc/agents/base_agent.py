"""辩论 Agent 抽象基类（V3 6.1 / 设计报告 4.2）。

所有辩论 Agent（提出方 / 批判方 / 异构）共享：
- 生成后端 LLMGenerator（依赖方向：agents → models，仅类型引用，运行时鸭子类型）；
- 采样参数：generation_kwargs 覆盖默认值（max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True）；
- 输出解析：正则优先匹配 "Label: <word>" 与 "Confidence: <num>"，
  Label 未命中时回退 utils.post_process.parse_label（首行截断 + 非法标签映射）。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..data.schemas import AgentOutput, DialogueContext
from ..utils.label_schema import LabelSchema, map_illegal_label
from ..utils.post_process import parse_label

if TYPE_CHECKING:
    from ..models.backbone import LLMGenerator

_LABEL_RE = re.compile(r"Label:\s*(\w+)")
_CONFIDENCE_RE = re.compile(r"Confidence:\s*([0-9.]+)")

DEFAULT_GENERATION_KWARGS: dict = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


class BaseAgent(ABC):
    """辩论 Agent 抽象基类：统一持有生成后端、标签表与输出解析逻辑。"""

    def __init__(
        self,
        generator: LLMGenerator,
        schema: LabelSchema,
        generation_kwargs: dict | None = None,
    ):
        self.generator = generator
        self.schema = schema
        self.generation_kwargs: dict = {**DEFAULT_GENERATION_KWARGS, **(generation_kwargs or {})}

    @abstractmethod
    def infer(self, context: DialogueContext, reasoning_hint: str | None = None) -> AgentOutput:
        """对目标话语做一次情感推断，返回结构化 AgentOutput。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 共享解析辅助
    # ------------------------------------------------------------------

    def _parse_label_and_confidence(self, text: str) -> tuple[str, float]:
        """从生成文本解析 (label, confidence)。

        - 正则匹配 `Label: <word>`；命中后仍做非法标签映射（保证落在 schema 内）
        - Label 未命中时用 utils.post_process.parse_label 解析首行
        - Confidence 未命中或解析失败时缺省 0.5；
          模型输出 0-100 量纲（>1 且 ≤100）时除以 100 归一，
          >100 视为无效回退 0.5；最终 clamp 到 [0, 1]
        """
        text = text or ""
        label_match = _LABEL_RE.search(text)
        if label_match:
            label = map_illegal_label(label_match.group(1), self.schema)
        else:
            label = parse_label(text, self.schema)

        confidence = 0.5
        conf_match = _CONFIDENCE_RE.search(text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5
            else:
                if 1.0 < confidence <= 100.0:
                    confidence = confidence / 100.0  # 0-100 量纲 → [0, 1]
                elif confidence > 100.0:
                    confidence = 0.5  # 无效置信度回退
        return label, min(1.0, max(0.0, confidence))

    def _generate(self, prompt: str) -> str:
        """按 generation_kwargs 调用生成后端（子类共用）。"""
        return self.generator.generate(prompt, **self.generation_kwargs)

    def _inject_hint(self, prompt: str, reasoning_hint: str | None) -> str:
        """将推理指引注入提示（模板未提供 hint 槽位时的统一兜底）。

        提示以 "Answer:" 结尾时插入其前，否则追加到末尾。
        """
        if not reasoning_hint:
            return prompt
        marker = "Answer:"
        stripped = prompt.rstrip()
        if stripped.endswith(marker):
            head = stripped[: -len(marker)].rstrip()
            return f"{head}\nReasoning guidance:\n{reasoning_hint}\n\n{marker}"
        return f"{prompt}\nReasoning guidance:\n{reasoning_hint}\n"
