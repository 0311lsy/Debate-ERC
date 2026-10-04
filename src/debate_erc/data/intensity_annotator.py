"""情绪强度自动标注（V3 6.8.4）：强/中/弱 三档派生。

优先使用 VADER（若安装）；否则回退到内置情感词强度词典。
三档切分阈值：compound ≥ 0.5 或 ≤ -0.5 → strong；
|compound| ∈ [0.25, 0.5) → medium；其余 → weak。
"""

from __future__ import annotations

from .schemas import DialogueSample

INTENSITY_LABELS = ("strong", "medium", "weak")

try:  # 可选依赖
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _Vader
    _VADER = _Vader()
except Exception:  # pragma: no cover - 环境无 vader 时回退
    _VADER = None

# 内置回退词典（compact VADER 子集 + 情感强度粗分）
_LEXICON: dict[str, float] = {
    "love": 3.0, "amazing": 3.0, "incredible": 3.0, "wonderful": 3.0, "wow": 3.0,
    "hate": -3.0, "horrible": -3.0, "terrible": -3.0, "disgusting": -3.0, "awful": -3.0,
    "great": 2.0, "happy": 2.0, "glad": 2.0, "excited": 2.5, "fun": 2.0, "thanks": 1.5,
    "good": 1.5, "nice": 1.5, "funny": 2.0, "sorry": -1.5, "sad": -2.0, "upset": -2.0,
    "angry": -2.5, "mad": -2.0, "scared": -2.5, "afraid": -2.5, "worried": -1.5,
    "damn": -2.0, "god": 0.5, "yeah": 0.5, "oh": 0.0, "fine": 0.5, "okay": 0.3,
    "never": -0.5, "really": 0.5, "so": 0.5, "what": -0.5, "no": -0.5,
}

_NEGATIONS = {"not", "no", "never", "don't", "dont", "didn't", "didnt", "isn't", "isnt"}
_BOOSTERS = {"very": 1.5, "so": 1.3, "really": 1.3, "totally": 1.5, "absolutely": 1.6}


def _fallback_score(text: str) -> float:
    """词典打分：否定翻转 + 程度词放大（booster 倍率作用于下一个命中情感词后重置），归一到 [-1, 1]。"""
    tokens = [t.lower().strip(".,!?;:'\"") for t in text.split()]
    total, hits = 0.0, 0
    negate = False
    booster = 1.0
    for tok in tokens:
        if tok in _NEGATIONS:
            negate = True
            continue
        if tok in _BOOSTERS:
            booster = _BOOSTERS[tok]
            continue
        if tok in _LEXICON:
            val = _LEXICON[tok] * booster
            booster = 1.0  # 放大仅作用于下一个命中情感词，随即重置
            if negate:
                val = -val * 0.75
            total += val
            hits += 1
            negate = False
    if hits == 0:
        return 0.0
    score = total / max(hits, 1) / 3.0
    return max(-1.0, min(1.0, score))


def intensity_compound(text: str) -> float:
    """返回 [-1, 1] 情感强度分（VADER compound 或回退词典分）。"""
    if _VADER is not None:
        return float(_VADER.polarity_scores(text)["compound"])
    return _fallback_score(text)


def intensity_label(text: str) -> str:
    """三档：strong / medium / weak。"""
    score = abs(intensity_compound(text))
    if score >= 0.5:
        return "strong"
    if score >= 0.25:
        return "medium"
    return "weak"


def annotate_intensity(samples: list[DialogueSample]) -> list[tuple[DialogueSample, str]]:
    """为样本批量派生强度标签（训练辅助任务用）。"""
    return [(sample, intensity_label(sample.context.target_utterance)) for sample in samples]
