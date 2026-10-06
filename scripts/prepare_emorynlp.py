#!/usr/bin/env python
"""EmoryNLP 原始 pkl → 项目 raw json（与 meld/iemocap raw 同构）。

输入：InstructERC/original_data/EmoryNLP/EmoryNLP.pkl
  data[0] dict[conv_id] -> 说话人 one-hot（9 个主要角色槽，InstructERC 口径）
  data[1] dict[conv_id] -> 情绪整数标签，顺序：
            0 Joyful, 1 Mad, 2 Peaceful, 3 Neutral, 4 Sad, 5 Powerful, 6 Scared
  data[2] dict[conv_id] -> 句子列表（= emorynlp_sentences.pkl）
  data[3/4/5] -> train/test/valid 对话 ID 列表（官方切分）

输出（默认 PRC-Emo/data/raw，供 --raw_path 直接读取）：
  emorynlp.train.json / emorynlp.valid.json / emorynlp.test.json
  结构 dict[conv_id] -> {"labels": [int], "sentences": [str], "speakers": [str]}

官方口径（Zahiri & Choi 2018 / InstructERC 复现）：
  train 7551（659 对话）/ valid 954（89）/ test 984（79）。
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

EMOTION_NAMES = ["joyful", "mad", "peaceful", "neutral", "sad", "powerful", "scared"]
SPLIT_IDS = {3: "train", 4: "test", 5: "valid"}
EXPECTED = {"train": 7551, "valid": 954, "test": 984}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/home/lsy20252770/InstructERC/original_data/EmoryNLP/EmoryNLP.pkl")
    ap.add_argument("--out", default="/home/lsy20252770/Agent_Reason/PRC-Emo/data/raw")
    args = ap.parse_args()

    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)
    speakers, labels, sentences = data[0], data[1], data[2]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, split in SPLIT_IDS.items():
        conv_ids = data[idx]
        payload = {}
        dist = Counter()
        n_utt = 0
        for cid in conv_ids:
            ys = [int(y) for y in labels[cid]]
            ss = [str(s).strip() for s in sentences[cid]]
            sp = [f"Speaker_{oh.index(1)}" for oh in speakers[cid]]
            assert len(ys) == len(ss) == len(sp), f"{cid} 三列表长度不一致"
            payload[cid] = {"labels": ys, "sentences": ss, "speakers": sp}
            dist.update(EMOTION_NAMES[y] for y in ys)
            n_utt += len(ys)
        target = out_dir / f"emorynlp.{split}.json"
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        assert n_utt == EXPECTED[split], f"{split} n={n_utt} 与官方 {EXPECTED[split]} 不符"
        print(f"{split:5s} {target}  对话={len(conv_ids)} 话语={n_utt}  分布={dict(dist)}")
    print("EmoryNLP raw json 转换完成，规模与官方口径一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
