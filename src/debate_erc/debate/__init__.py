"""辩论包（V3 6.1/6.2/6.7）：主循环、收敛判定、Re-ask 兜底与共识统计。

依赖方向：debate → agents（仅类型）/ data / utils；不得 import training。
"""

from __future__ import annotations

from .consensus import summarize_debate
from .convergence import ConvergenceChecker
from .debate_loop import DebateLoop
from .reask_fallback import ReaskFallback

__all__ = [
    "DebateLoop",
    "ConvergenceChecker",
    "ReaskFallback",
    "summarize_debate",
]
