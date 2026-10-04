"""辅助任务包（V3 6.8 / 设计报告 2.1）：强度估计 + 说话人归属 + 多任务调度。

依赖方向：aux_task → data / models / training（经 trainer 注入，不反向依赖 pipeline）。
"""

from __future__ import annotations

from .intensity_head import (
    INTENSITY_SCHEMA,
    IntensityEstimator,
    SpeakerEstimator,
    build_intensity_prompt,
    build_speaker_prompt,
)
from .multitask_scheduler import MultitaskScheduler

__all__ = [
    "INTENSITY_SCHEMA",
    "IntensityEstimator",
    "SpeakerEstimator",
    "build_intensity_prompt",
    "build_speaker_prompt",
    "MultitaskScheduler",
]
