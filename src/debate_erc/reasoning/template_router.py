"""错误模式 → 推理模板路由（V3 7.1）。

按错误模式特征选择模板（先匹配先用）：
1. LONG_UTTERANCE：目标话语长度 ≥ long_utterance_chars（默认 150）字符
   （与 DifficultyRouter 的 ≥ 口径一致，R2-m3 统一边界语义）；
2. IRONY：目标含积极词但上下文含消极词，或反讽标记（"yeah right" / "great" /
   "thanks a lot" / "just perfect" 等）配合上下文否定；
3. MINORITY：含 fear / disgust 词典词（scared, afraid, terrified, gross, disgusting, sick 等），
   仅当 schema 标签表含 "fear" 时可路由（如 IEMOCAP 4 类无 fear），否则回退 TRIGGER；
4. TRIGGER：有前文上下文（target_idx > 0）；
5. LABEL_CHECK：兜底。
"""

from __future__ import annotations

import re

from ..data.schemas import DialogueContext
from ..utils.label_schema import LabelSchema
from .templates import ReasoningTemplates, TemplateType


def _compile_word_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    """词边界匹配模式（避免 'another' 误命中 'not' 这类子串误报）。"""
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in words))


class TemplateRouter:
    """简单词表启发式的模板路由器。"""

    IRONY_MARKERS: tuple[str, ...] = (
        "yeah right", "thanks a lot", "just perfect", "oh great",
        "great", "perfect", "wonderful",
    )
    POSITIVE_WORDS: tuple[str, ...] = (
        "great", "good", "wonderful", "perfect", "nice", "love", "loved",
        "thanks", "thank", "congratulations", "awesome", "fantastic",
    )
    NEGATIVE_WORDS: tuple[str, ...] = (
        "bad", "terrible", "awful", "hate", "angry", "mad", "sad", "wrong",
        "stupid", "broken", "late", "lost", "never mind", "forget it",
    )
    NEGATION_WORDS: tuple[str, ...] = (
        "not", "never", "no way", "don't", "didn't", "can't", "won't",
    )
    MINORITY_LEXICON: tuple[str, ...] = (
        "scared", "afraid", "terrified", "fear", "fearful", "frightened",
        "horrified", "gross", "disgusting", "disgusted", "sick",
    )

    _IRONY_MARKER_RE = _compile_word_pattern(IRONY_MARKERS)
    _POSITIVE_RE = _compile_word_pattern(POSITIVE_WORDS)
    _NEGATIVE_RE = _compile_word_pattern(NEGATIVE_WORDS)
    _NEGATION_RE = _compile_word_pattern(NEGATION_WORDS)
    _MINORITY_RE = _compile_word_pattern(MINORITY_LEXICON)

    def __init__(self, schema: LabelSchema, long_utterance_chars: int = 150):
        """schema: 标签表（MINORITY 模板要求 "fear" 在标签表内）；
        long_utterance_chars: 长话语判定的字符阈值（assembler 注入 routing 配置）。
        """
        self.schema = schema
        self.long_utterance_chars = long_utterance_chars

    def route(self, context: DialogueContext) -> TemplateType:
        """按错误模式特征选择模板（优先级：长度 > 反讽 > 少数类 > 触发 > 兜底）。"""
        target = context.target_utterance.lower()
        history_text = " ".join(context.history).lower()

        if len(context.target_utterance) >= self.long_utterance_chars:
            return TemplateType.LONG_UTTERANCE
        if self._irony_cue(target, history_text):
            return TemplateType.IRONY
        if (
            "fear" in self.schema.labels
            and (self._MINORITY_RE.search(target) or self._MINORITY_RE.search(history_text))
        ):
            return TemplateType.MINORITY
        if context.target_idx > 0:
            return TemplateType.TRIGGER
        return TemplateType.LABEL_CHECK

    def build_hint(self, context: DialogueContext) -> str:
        """route 后返回对应模板文本（注入分类提示的推理指引）。"""
        return ReasoningTemplates.get(self.route(context))

    def _irony_cue(self, target_lower: str, history_lower: str) -> bool:
        """反讽线索：目标含积极词或反讽标记，且上下文含消极词 / 否定。"""
        if not history_lower:
            return False
        context_negative = bool(
            self._NEGATIVE_RE.search(history_lower)
            or self._NEGATION_RE.search(history_lower)
        )
        if not context_negative:
            return False
        return bool(
            self._POSITIVE_RE.search(target_lower)
            or self._IRONY_MARKER_RE.search(target_lower)
        )
