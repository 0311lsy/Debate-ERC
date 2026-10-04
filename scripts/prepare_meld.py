#!/usr/bin/env python
"""数据准备与官方类目核对（验收标准 0.2-①：逐项相等，防 V2 类目错误复现）。

用法：
    python scripts/prepare_meld.py [--raw-path /path/to/raw] [--dataset meld|iemocap]
                                   [--out data/processed]

产出：<out>/<dataset>_stats.json（三 split 类目分布，与官方口径逐项核对）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.data import load_split  # noqa: E402
from debate_erc.data.meld_loader import check_official_counts  # noqa: E402
from debate_erc.data.iemocap_loader import label_distribution  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.prepare")


def main() -> int:
    parser = argparse.ArgumentParser(description="MELD/IEMOCAP 数据核对与统计落盘")
    parser.add_argument("--raw-path", default="/home/lsy20252770/Agent_Reason/PRC-Emo/data/raw")
    parser.add_argument("--dataset", default="meld", choices=["meld", "iemocap"])
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed"))
    args = parser.parse_args()

    stats: dict = {"dataset": args.dataset, "splits": {}}
    for split in ("train", "dev", "test"):
        samples = load_split(args.dataset, args.raw_path, split, args.history_window)
        if args.dataset == "meld":
            dist = check_official_counts(samples, split)  # 不符直接 AssertionError
        else:
            dist = label_distribution(samples)
        stats["splits"][split] = {"n": len(samples), "distribution": dist}
        logger.info("[%s] %s: n=%d %s", args.dataset, split, len(samples), dist)

    out_path = Path(args.out) / f"{args.dataset}_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info("类目核对全部通过，统计已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
