"""流水线层（Layer 5 顶层）：装配、编排与 CLI 入口。

依赖方向：pipeline → 所有层；不含业务逻辑，只做组装与调度（设计报告 2.1 #11）。
"""

from __future__ import annotations

from .assembler import PipelineAssembler, Runtime, build_runtime
from .runner import ExperimentRunner
from .stages import (
    DebateStage,
    InferenceState,
    LabelStage,
    PipelineStage,
    ReasoningStage,
    RouteStage,
    run_pipeline,
)

__all__ = [
    "PipelineAssembler",
    "Runtime",
    "build_runtime",
    "ExperimentRunner",
    "PipelineStage",
    "InferenceState",
    "RouteStage",
    "ReasoningStage",
    "DebateStage",
    "LabelStage",
    "run_pipeline",
]
