"""DailyDialog 加载器（M2(a) 域外泛化实验）。

数据来源：HF daily_dialog（parquet 转换分支），CC BY-NC-SA 4.0。
标签映射到 MELD 7 类口径（仅 no emotion→neutral / happiness→joyful 改名，其余同名）。
无说话人标注：交替分配 A/B。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .schemas import DialogueContext, DialogueSample

DD_INDEX_TO_LABEL = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "joyful", 5: "sadness", 6: "surprise",
}

DEFAULT_RAW_SPLITS = {
    "train": "daily_dialog/train.parquet",
    "dev": "daily_dialog/validation.parquet",
    "test": "daily_dialog/test.parquet",
}


def load_dailydialog_split(
    raw_path: str | Path, split: str, history_window: int = 5
) -> list[DialogueSample]:
    path = Path(raw_path)
    if path.is_dir():
        path = path / DEFAULT_RAW_SPLITS[split]
    df = pd.read_parquet(path)
    samples: list[DialogueSample] = []
    for dia_idx, row in df.iterrows():
        utterances = [u.strip() for u in row["dialog"]]
        emotions = row["emotion"]
        speakers = [("A" if i % 2 == 0 else "B") for i in range(len(utterances))]
        if len(utterances) != len(emotions):
            raise ValueError(
                f"DailyDialog dialogue {dia_idx} 字段长度不一致: "
                f"utterances={len(utterances)} emotions={len(emotions)}"
            )
        for idx, emo_id in enumerate(emotions):
            start = max(0, idx - history_window)
            context = DialogueContext(
                utterances=tuple(utterances[start : idx + 1]),
                speakers=tuple(speakers[start : idx + 1]),
                target_idx=idx - start,
                history_window=history_window,
                dialogue_id=f"dd_{dia_idx}",
            )
            samples.append(DialogueSample(
                context=context,
                gold_label=DD_INDEX_TO_LABEL[int(emo_id)],
                synthetic=False,
                split=split,
            ))
    return samples


def dailydialog_stats(samples: list[DialogueSample]) -> dict[str, int]:
    return dict(Counter(s.gold_label for s in samples))
