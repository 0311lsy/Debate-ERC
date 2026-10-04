"""推理指引包（V3 7.1）：模板库 + 错误模式路由。

依赖方向：reasoning → data / utils。
"""

from __future__ import annotations

from .template_router import TemplateRouter
from .templates import ReasoningTemplates, TemplateType

__all__ = [
    "TemplateType",
    "ReasoningTemplates",
    "TemplateRouter",
]
