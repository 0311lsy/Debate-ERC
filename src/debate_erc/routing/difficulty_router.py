"""难度路由（V3 7.2）：PERCEIVE / COMPOSE / DELIBERATE 三级路由。

特征 = 目标话语长度 + 说话人切换数 + 上下文轮数。规则：
- 长度 ≤ short_threshold 且无说话人切换且上下文 ≤ 2 轮 → PERCEIVE（轻量：单轮感知）；
- 长度 ≥ long_threshold 或含少数类情感词（minority_boost）或说话人切换 ≥ 3 →
  DELIBERATE（重量：完整辩论）；
- 其余 → COMPOSE（中等：结构化推理）。
"""

from __future__ import annotations

import enum
import re

from ..data.schemas import DialogueContext

DEFAULT_MINORITY_LEXICON: frozenset[str] = frozenset(
    {
        "scared", "afraid", "terrified", "fear", "fearful", "frightened",
        "horrified", "panic", "gross", "disgusting", "disgusted", "disgust",
        "sick",
    }
)


class RouteMode(enum.Enum):
    """三级推理模式（V3 7.2）。"""

    PERCEIVE = "perceive"      # 轻量：单 Agent 直接分类
    COMPOSE = "compose"        # 中等：结构化推理（推理指引）
    DELIBERATE = "deliberate"  # 重量：完整多轮辩论


class DifficultyRouter:
    """基于简单特征的难度路由器。"""

    def __init__(
        self,
        short_threshold: int = 60,
        long_threshold: int = 150,
        minority_boost: bool = True,
        minority_lexicon: set[str] | None = None,
    ):
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.minority_boost = minority_boost
        self.minority_lexicon: frozenset[str] = (
            frozenset(minority_lexicon)
            if minority_lexicon is not None
            else DEFAULT_MINORITY_LEXICON
        )
        self._minority_re: re.Pattern[str] | None = (
            re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in sorted(self.minority_lexicon)))
            if self.minority_lexicon
            else None
        )

    def route(self, context: DialogueContext) -> RouteMode:
        """单样本路由：话语长度 + 说话人切换数 + 上下文轮数。"""
        length = len(context.target_utterance)
        switches = self._speaker_switches(context)
        context_turns = context.target_idx  # 前文轮数

        if length <= self.short_threshold and switches == 0 and context_turns <= 2:
            return RouteMode.PERCEIVE
        if (
            length >= self.long_threshold
            or (self.minority_boost and self._has_minority_cue(context))
            or switches >= 3
        ):
            return RouteMode.DELIBERATE
        return RouteMode.COMPOSE

    def route_batch(self, contexts: list[DialogueContext]) -> list[RouteMode]:
        """批量路由。"""
        return [self.route(c) for c in contexts]

    def distribution(self, modes: list[RouteMode]) -> dict[str, int]:
        """模式分布计数（key 为模式名，三种模式均出现，缺省 0）。"""
        counts = {mode.value: 0 for mode in RouteMode}
        for mode in modes:
            counts[mode.value] += 1
        return counts

    @staticmethod
    def _speaker_switches(context: DialogueContext) -> int:
        """说话人切换数：相邻话语说话人不同的次数。"""
        speakers = context.speakers
        return sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i - 1])

    def _has_minority_cue(self, context: DialogueContext) -> bool:
        """目标话语或前文含少数类情感词（fear / disgust 词典）。"""
        if self._minority_re is None:
            return False
        text = (context.target_utterance + " " + " ".join(context.history)).lower()
        return bool(self._minority_re.search(text))
