"""MELD 数据加载器：官方 7 类口径（V3 4.2 类目表）。

原始数据为 PRC-Emo 归档的 raw json（dict[dialogue_id] -> {labels, sentences, speakers}），
标签索引映射：{0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..utils.label_schema import MELD_OFFICIAL_COUNTS, MELD_SCHEMA
from .schemas import DialogueContext, DialogueSample

# PRC-Emo raw json 的标签索引 → 本项目统一标签名（joy → joyful）
MELD_INDEX_TO_LABEL = {
    0: "neutral", 1: "surprise", 2: "fear", 3: "sadness",
    4: "joyful", 5: "disgust", 6: "anger",
}

DEFAULT_RAW_SPLITS = {
    "train": "meld.train.json",
    "dev": "meld.valid.json",
    "test": "meld.test.json",
}


def load_meld_split(raw_path: str | Path, split: str, history_window: int = 5) -> list[DialogueSample]:
    """加载一个 MELD split，返回 DialogueSample 列表（每条话语一个样本）。"""
    path = Path(raw_path)
    if path.is_dir():
        path = path / DEFAULT_RAW_SPLITS[split]
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    samples: list[DialogueSample] = []
    for dialogue_id, dialogue in data.items():
        sentences = [s.strip() for s in dialogue["sentences"]]
        speakers = [str(s) for s in dialogue.get("speakers", ["A"] * len(sentences))]
        labels = dialogue["labels"]
        # 三并行列表长度一致性校验（R2-m14）：不一致时静默 zip 截断/错位会污染
        # 训练数据且难以察觉，加载期显式失败
        if len(speakers) != len(sentences) or len(labels) != len(sentences):
            raise ValueError(
                f"MELD dialogue {dialogue_id} 字段长度不一致: "
                f"sentences={len(sentences)} labels={len(labels)} speakers={len(speakers)}"
            )
        for idx, label_idx in enumerate(labels):
            # 上下文 = 目标话语前 history_window 轮 + 目标话语
            start = max(0, idx - history_window)
            context = DialogueContext(
                utterances=tuple(sentences[start : idx + 1]),
                speakers=tuple(speakers[start : idx + 1]),
                target_idx=idx - start,
                history_window=history_window,
                dialogue_id=str(dialogue_id),
            )
            samples.append(DialogueSample(
                context=context,
                gold_label=MELD_INDEX_TO_LABEL[int(label_idx)],
                synthetic=False,
                split=split,
            ))
    return samples


def check_official_counts(samples: list[DialogueSample], split: str) -> dict[str, int]:
    """与官方类目逐项核对（验收标准 0.2-①，防 V2 类目错误复现）。返回实际分布。"""
    actual = Counter(s.gold_label for s in samples)
    expected = MELD_OFFICIAL_COUNTS[split]
    actual_dict = {label: actual.get(label, 0) for label in MELD_SCHEMA.labels}
    if actual_dict != expected:
        diff = {k: (actual_dict[k], expected[k]) for k in expected if actual_dict[k] != expected[k]}
        raise AssertionError(
            f"MELD {split} 类目与官方不符 (actual, expected): {diff}"
        )
    return actual_dict
