"""标签体系：MELD 7 类 / IEMOCAP 4 类常量与非法标签余弦映射。

V3 4.4 评估协议：非法标签余弦映射到最近合法标签。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# MELD 官方 7 类（V3 4.2 类目表口径）
MELD_LABELS: tuple[str, ...] = (
    "neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear",
)

# IEMOCAP 10→4 映射（V3 4.1，4 类口径：neutral/joyful/sadness/anger；
# raw json 的 happy/excited → joyful，frustrated/angry → anger）
IEMOCAP_LABELS: tuple[str, ...] = ("neutral", "joyful", "sadness", "anger")
IEMOCAP_LABEL_MAP: dict[str, str] = {
    "happy": "joyful",
    "excited": "joyful",
    "sad": "sadness",
    "angry": "anger",
    "frustrated": "anger",
    "neutral": "neutral",
}

# EmoryNLP（EmotionLines / Friends，Zahiri & Choi 2018）官方 7 类，与 MELD 标签
# 空间不同（peaceful/powerful/mad/scared），故不做跨数据集门控迁移，只做域内协议复现
EMORYNLP_LABELS: tuple[str, ...] = (
    "neutral", "joyful", "mad", "peaceful", "sad", "powerful", "scared",
)
EMORYNLP_OFFICIAL_COUNTS: dict[str, dict[str, int]] = {
    "train": {"neutral": 2485, "joyful": 1677, "mad": 785, "peaceful": 638,
              "sad": 474, "powerful": 551, "scared": 941},
    "dev": {"neutral": 322, "joyful": 205, "mad": 97, "peaceful": 82,
            "sad": 51, "powerful": 70, "scared": 127},
    "test": {"neutral": 288, "joyful": 217, "mad": 86, "peaceful": 111,
             "sad": 70, "powerful": 96, "scared": 116},
}

# MELD 官方类目数（train/dev/test），用于 tests/test_data_official_counts.py 逐项核对
MELD_OFFICIAL_COUNTS: dict[str, dict[str, int]] = {
    "train": {
        "neutral": 4710, "joyful": 1743, "surprise": 1205, "anger": 1109,
        "sadness": 683, "disgust": 271, "fear": 268,
    },
    "dev": {
        "neutral": 470, "joyful": 163, "surprise": 150, "anger": 153,
        "sadness": 111, "disgust": 22, "fear": 40,
    },
    "test": {
        "neutral": 1256, "joyful": 402, "surprise": 281, "anger": 345,
        "sadness": 208, "disgust": 68, "fear": 50,
    },
}


@dataclass(frozen=True)
class LabelSchema:
    """单一事实来源：一个数据集一套标签表。"""

    labels: tuple[str, ...]
    name: str = "meld"

    @property
    def size(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        return self.labels.index(label)

    def is_legal(self, label: str) -> bool:
        return label in self.labels


MELD_SCHEMA = LabelSchema(MELD_LABELS, "meld")
IEMOCAP_SCHEMA = LabelSchema(IEMOCAP_LABELS, "iemocap")
EMORYNLP_SCHEMA = LabelSchema(EMORYNLP_LABELS, "emorynlp")

_SCHEMAS: dict[str, LabelSchema] = {
    "meld": MELD_SCHEMA,
    "iemocap": IEMOCAP_SCHEMA,
    "emorynlp": EMORYNLP_SCHEMA,
    "dailydialog": MELD_SCHEMA,
}


def get_schema(name: str) -> LabelSchema:
    if name not in _SCHEMAS:
        raise KeyError(f"未知数据集 schema: {name}，可选 {list(_SCHEMAS)}")
    return _SCHEMAS[name]


def _char_ngram_cosine(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram 余弦相似度（标签词都很短，n=2 足够）。"""
    def grams(s: str) -> Counter:
        s = f"^{s}$"
        return Counter(s[i : i + n] for i in range(len(s) - n + 1)) if len(s) > n else Counter({s: 1})

    ga, gb = grams(a.lower()), grams(b.lower())
    if not ga or not gb:
        return 0.0
    dot = sum(ga[g] * gb[g] for g in ga if g in gb)
    norm = math.sqrt(sum(v * v for v in ga.values())) * math.sqrt(sum(v * v for v in gb.values()))
    return dot / norm if norm else 0.0


def map_illegal_label(raw: str, schema: LabelSchema = MELD_SCHEMA) -> str:
    """非法标签 → 最近合法标签。

    规则（按优先级）：
    1. 完全匹配 / 大小写归一后匹配
    2. 词干归一（去尾部 'ed'/'s' 等屈折，如 angry→anger, joys→joyful 不适用则走 3）
    3. 子串包含（如 "anger issue" → anger）
    4. 字符 n-gram 余弦最近邻（如 "angr" → anger, "suprised" → surprise）
    """
    raw = (raw or "").strip().lower()
    if not raw:
        return schema.labels[0]
    if raw in schema.labels:
        return raw
    # 词干归一
    stem_map = {"angry": "anger", "mad": "anger", "joy": "joyful", "happy": "joyful",
                "sad": "sadness", "surprised": "surprise", "fearful": "fear",
                "disgusted": "disgust", "frustrated": "anger", "excited": "joyful"}
    if raw in stem_map and stem_map[raw] in schema.labels:
        return stem_map[raw]
    # 子串包含
    for label in schema.labels:
        if label in raw or raw in label:
            return label
    # 余弦最近邻
    return max(schema.labels, key=lambda lb: _char_ngram_cosine(raw, lb))


_LABEL_PATTERN = re.compile(r"[a-zA-Z]+")


def extract_label_token(text: str) -> str | None:
    """从一行输出中提取候选标签词（第一个纯字母 token）。"""
    for tok in _LABEL_PATTERN.findall(text or ""):
        return tok
    return None
