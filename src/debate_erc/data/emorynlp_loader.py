"""EmoryNLP（EmotionLines, Friends）数据加载器：官方 7 类口径。

raw json 由 scripts/prepare_emorynlp.py 从 InstructERC 归档 pkl 转出，
标签索引：0 joyful, 1 mad, 2 peaceful, 3 neutral, 4 sad, 5 powerful, 6 scared。
与 MELD 标签空间不同，跨数据集不直接比较、不做门控迁移。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..utils.label_schema import EMORYNLP_OFFICIAL_COUNTS, EMORYNLP_SCHEMA
from .schemas import DialogueContext, DialogueSample

EMORYNLP_INDEX_TO_LABEL = {
    0: "joyful", 1: "mad", 2: "peaceful", 3: "neutral",
    4: "sad", 5: "powerful", 6: "scared",
}

DEFAULT_RAW_SPLITS = {
    "train": "emorynlp.train.json",
    "dev": "emorynlp.valid.json",
    "test": "emorynlp.test.json",
}


def load_emorynlp_split(raw_path: str | Path, split: str, history_window: int = 5) -> list[DialogueSample]:
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
        if len(speakers) != len(sentences) or len(labels) != len(sentences):
            raise ValueError(
                f"EmoryNLP dialogue {dialogue_id} 字段长度不一致: "
                f"sentences={len(sentences)} labels={len(labels)} speakers={len(speakers)}"
            )
        for idx, label_idx in enumerate(labels):
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
                gold_label=EMORYNLP_INDEX_TO_LABEL[int(label_idx)],
                synthetic=False,
                split=split,
            ))
    return samples


def check_official_counts(samples: list[DialogueSample], split: str) -> dict[str, int]:
    actual = Counter(s.gold_label for s in samples)
    expected = EMORYNLP_OFFICIAL_COUNTS[split]
    actual_dict = {label: actual.get(label, 0) for label in EMORYNLP_SCHEMA.labels}
    if actual_dict != expected:
        diff = {k: (actual_dict[k], expected[k]) for k in expected if actual_dict[k] != expected[k]}
        raise AssertionError(f"EmoryNLP {split} 类目与官方不符 (actual, expected): {diff}")
    return actual_dict
