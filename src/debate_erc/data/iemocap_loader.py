"""IEMOCAP 数据加载器：10→4 类映射（V3 4.1 映射表驱动）。

PRC-Emo 归档 raw json 仅含 6 类（happy/sad/neutral/angry/excited/frustrated），
经 IEMOCAP_LABEL_MAP 映射到本项目 4 类口径（happy/excited → joyful、
frustrated/angry → anger、sad → sadness、neutral → neutral）；
跨数据集不与 MELD 直接比较。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..utils.label_schema import IEMOCAP_LABEL_MAP, IEMOCAP_SCHEMA
from .schemas import DialogueContext, DialogueSample

# PRC-Emo iemocap raw json 标签索引（与 reformat_data_ft_llm_combine.py 一致）
IEMOCAP_INDEX_TO_RAW = {0: "happy", 1: "sad", 2: "neutral", 3: "angry", 4: "excited", 5: "frustrated"}

DEFAULT_RAW_SPLITS = {
    "train": "iemocap.train.json",
    "dev": "iemocap.valid.json",
    "test": "iemocap.test.json",
}


def load_iemocap_split(raw_path: str | Path, split: str, history_window: int = 5) -> list[DialogueSample]:
    path = Path(raw_path)
    if path.is_dir():
        path = path / DEFAULT_RAW_SPLITS[split]
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    samples: list[DialogueSample] = []
    for dialogue_id, dialogue in data.items():
        sentences = [s.strip() for s in dialogue["sentences"]]
        speakers = [str(s) for s in dialogue.get("speakers", ["A"] * len(sentences))]
        for idx, label_idx in enumerate(dialogue["labels"]):
            raw_label = IEMOCAP_INDEX_TO_RAW[int(label_idx)]
            mapped = IEMOCAP_LABEL_MAP.get(raw_label, "neutral")
            start = max(0, idx - history_window)
            context = DialogueContext(
                utterances=tuple(sentences[start : idx + 1]),
                speakers=tuple(speakers[start : idx + 1]),
                target_idx=idx - start,
                history_window=history_window,
                dialogue_id=str(dialogue_id),
            )
            samples.append(DialogueSample(
                context=context, gold_label=mapped, synthetic=False, split=split,
            ))
    return samples


def label_distribution(samples: list[DialogueSample]) -> dict[str, int]:
    cnt = Counter(s.gold_label for s in samples)
    return {label: cnt.get(label, 0) for label in IEMOCAP_SCHEMA.labels}
