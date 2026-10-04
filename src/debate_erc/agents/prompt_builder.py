"""提示构建：基础分类提示（base_sft / LabelStage 共用）。

模板文件位于 agents/prompts/（jinja2，与代码分离）。
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ..utils.label_schema import LabelSchema
from ..data.schemas import DialogueContext

PROMPT_DIR = Path(__file__).parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(PROMPT_DIR)),
    autoescape=select_autoescape(default=False),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render(template_name: str, **kwargs) -> str:
    template = _env.get_template(template_name)
    return template.render(**kwargs)


def build_classify_prompt(
    context: DialogueContext,
    schema: LabelSchema,
    instruction: str | None = None,
    reasoning_hint: str | None = None,
) -> str:
    """基础情感分类提示（单 Agent 直接推理 / SFT 训练目标构建共用）。"""
    return render(
        "classify.j2",
        dialogue=context.render(),
        target_speaker=context.target_speaker,
        target_utterance=context.target_utterance,
        labels=list(schema.labels),
        instruction=instruction,
        reasoning_hint=reasoning_hint,
    )
