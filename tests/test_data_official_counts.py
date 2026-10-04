"""MELD 官方类目核对测试（审查修复 E 任务 7，验收标准 0.2-①）。

不用真实 raw 数据（依赖外部路径），而是按 MELD_OFFICIAL_COUNTS 的数量与
标签分布**伪造** DialogueSample（全量 9989/1109/2610 条，dataclass 构造毫秒级），
对拍 meld_loader.check_official_counts 的判定逻辑：
- 官方分布的伪造样本 → 通过且返回分布逐项相等；
- 缺失标签 / 数量偏差 / 非法标签名 → AssertionError（防 V2 类目错误复现）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.data.meld_loader import (  # noqa: E402
    MELD_INDEX_TO_LABEL,
    check_official_counts,
)
from debate_erc.data.schemas import DialogueContext, DialogueSample  # noqa: E402
from debate_erc.utils.label_schema import (  # noqa: E402
    MELD_OFFICIAL_COUNTS,
    MELD_SCHEMA,
)

SPLITS = ("train", "dev", "test")
# MELD 官方 split 总数（V3 4.2 类目表口径）
OFFICIAL_TOTALS = {"train": 9989, "dev": 1109, "test": 2610}


def make_fake_samples(
    split: str,
    counts: dict[str, int] | None = None,
) -> list[DialogueSample]:
    """按给定类目分布伪造 DialogueSample（gold_label 驱动，上下文最简）。"""
    counts = dict(counts if counts is not None else MELD_OFFICIAL_COUNTS[split])
    samples: list[DialogueSample] = []
    for label, n in counts.items():
        for i in range(n):
            samples.append(DialogueSample(
                context=DialogueContext(
                    utterances=(f"utt {label} {i}",),
                    speakers=("A",),
                    target_idx=0,
                    history_window=0,
                    dialogue_id=f"{split}_{label}_{i}",
                ),
                gold_label=label,
                synthetic=False,
                split=split,
            ))
    return samples


class TestOfficialCountsConstant:
    """MELD_OFFICIAL_COUNTS 常量自身口径（A/D 组注释指向的本测试）。"""

    @pytest.mark.parametrize("split", SPLITS)
    def test_totals_match_official_split_sizes(self, split):
        assert sum(MELD_OFFICIAL_COUNTS[split].values()) == OFFICIAL_TOTALS[split], (
            f"{split} 类目总数与官方 {OFFICIAL_TOTALS[split]} 不符"
        )

    @pytest.mark.parametrize("split", SPLITS)
    def test_labels_cover_schema_exactly(self, split):
        assert set(MELD_OFFICIAL_COUNTS[split]) == set(MELD_SCHEMA.labels), (
            f"{split} 类目键与 MELD_SCHEMA.labels 不一致"
        )

    def test_all_counts_positive(self):
        for split in SPLITS:
            for label, n in MELD_OFFICIAL_COUNTS[split].items():
                assert n > 0, f"{split}.{label} 计数须为正: {n}"

    def test_label_index_mapping_covers_schema(self):
        """raw 标签索引映射完整：MELD_INDEX_TO_LABEL 的值恰为 7 类合法标签。"""
        assert set(MELD_INDEX_TO_LABEL.values()) == set(MELD_SCHEMA.labels)
        assert sorted(MELD_INDEX_TO_LABEL) == list(range(7))


class TestCheckOfficialCountsAccepts:
    """按官方分布伪造的样本应逐项通过核对。"""

    @pytest.mark.parametrize("split", SPLITS)
    def test_fake_samples_pass_and_return_distribution(self, split):
        samples = make_fake_samples(split)
        dist = check_official_counts(samples, split)  # 不符会直接 AssertionError
        assert dist == MELD_OFFICIAL_COUNTS[split]
        assert sum(dist.values()) == len(samples)

    @pytest.mark.parametrize("split", SPLITS)
    def test_returned_distribution_keyed_by_schema_order(self, split):
        dist = check_official_counts(make_fake_samples(split), split)
        assert list(dist) == list(MELD_SCHEMA.labels)


class TestCheckOfficialCountsRejects:
    """类目偏离官方口径必须被拒绝（防 V2 类目错误复现）。"""

    def test_missing_label_class_rejected(self):
        counts = dict(MELD_OFFICIAL_COUNTS["dev"])
        counts.pop("fear")  # 整类缺失
        with pytest.raises(AssertionError, match="fear"):
            check_official_counts(make_fake_samples("dev", counts), "dev")

    def test_count_off_by_one_rejected(self):
        counts = dict(MELD_OFFICIAL_COUNTS["dev"])
        counts["joyful"] += 1
        samples = make_fake_samples("dev", counts)
        with pytest.raises(AssertionError, match="joyful"):
            check_official_counts(samples, "dev")

    def test_count_swapped_between_labels_rejected(self):
        """两类间数量互换（总数不变）也须被逐项核对拒绝。"""
        counts = dict(MELD_OFFICIAL_COUNTS["dev"])
        counts["disgust"], counts["fear"] = counts["fear"], counts["disgust"]
        with pytest.raises(AssertionError):
            check_official_counts(make_fake_samples("dev", counts), "dev")

    def test_illegal_label_name_rejected(self):
        """非法标签名（如 raw 口径 "happy" 未映射为 "joyful"）导致 joy 计数缺额。"""
        samples = make_fake_samples("dev")
        idx = next(
            i for i, s in enumerate(samples) if s.gold_label == "joyful"
        )
        samples[idx] = DialogueSample(
            context=samples[idx].context,
            gold_label="happy",  # 非法：未走 MELD_INDEX_TO_LABEL 映射
            synthetic=False,
            split="dev",
        )
        with pytest.raises(AssertionError, match="joyful"):
            check_official_counts(samples, "dev")

    def test_empty_sample_list_rejected(self):
        with pytest.raises(AssertionError):
            check_official_counts([], "train")

    def test_wrong_split_compared_rejected(self):
        """dev 分布的样本按 train 口径核对（split 错配）必须拒绝。"""
        with pytest.raises(AssertionError):
            check_official_counts(make_fake_samples("dev"), "train")
