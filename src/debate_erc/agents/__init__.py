"""辩论 Agent 包（V3 6.1）：提出方 / 批判方 / 异构 + 提示构建工具。

依赖方向：agents → models（仅类型）/ data / utils。
"""

from __future__ import annotations

from .base_agent import BaseAgent
from .causal_agent import CausalProponentAgent
from .critic_agent import CriticAgent
from .direct_agent import DirectClassifyAgent
from .hetero_agent import HeteroAgent
from .prompt_builder import build_classify_prompt, render

__all__ = [
    "BaseAgent",
    "CausalProponentAgent",
    "CriticAgent",
    "DirectClassifyAgent",
    "HeteroAgent",
    "build_classify_prompt",
    "render",
]
