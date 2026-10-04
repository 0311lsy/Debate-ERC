"""DPO 偏好对构建测试（审查修复 E 任务 5，S3 契约）。

GoldLabelPairBuilder 三条口径：
a) chosen/rejected 同构短格式 `f" {label}"`——答对 chosen=gold / rejected=混淆先验
   错误答案；答错 chosen=gold / rejected=模型自己的错误输出；
b) 不读取辩论收敛状态字段（converged=True/False 输出一致）；
c) final_reasoning 不进入偏好对（长推理链与短答案混排会破坏同构性）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.data.schemas import AgentOutput, DebateOutcome  # noqa: E402
from debate_erc.training import GoldLabelPairBuilder, build_pairs  # noqa: E402
from debate_erc.training.preference_pair import (  # noqa: E402
    DEFAULT_CONFUSION_PRIOR,
)
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402


def make_outcome(
    final_label: str,
    *,
    converged: bool = False,
    final_reasoning: str = "",
    reask_triggered: bool = False,
) -> DebateOutcome:
    """构造单 Agent 口径的最小 DebateOutcome（偏好对构造只需要 final_label）。"""
    return DebateOutcome(
        final_label=final_label,
        confidence=0.8,
        rounds=2,
        converged=converged,
        reask_triggered=reask_triggered,
        proponent_output=AgentOutput(label=final_label, confidence=0.8),
        final_reasoning=final_reasoning,
    )


SHORT_LABEL_FORMATS = [f" {lb}" for lb in MELD_SCHEMA.labels]


class TestChosenRejectedContract:
    """a) 答对 / 答错两类样本的 chosen/rejected 同构短格式。"""

    def test_correct_sample_gold_chosen_confusion_rejected(self):
        builder = GoldLabelPairBuilder()
        gold = "joyful"
        outcome = make_outcome(gold, converged=True)
        pair = builder.build(outcome, gold, prompt="ctx A: ...")
        # 答对：chosen = gold，rejected = 混淆先验的最常见误判类
        assert pair.chosen == f" {gold}"
        wrong = DEFAULT_CONFUSION_PRIOR[gold]
        assert wrong != gold
        assert pair.rejected == f" {wrong}"

    def test_wrong_sample_gold_chosen_model_label_rejected(self):
        builder = GoldLabelPairBuilder()
        gold, model_label = "sadness", "anger"
        assert model_label != gold
        outcome = make_outcome(model_label)
        pair = builder.build(outcome, gold, prompt="ctx A: ...")
        # 答错：chosen = gold（正确答案），rejected = 模型自己的错误输出
        assert pair.chosen == f" {gold}"
        assert pair.rejected == f" {model_label}"

    @pytest.mark.parametrize("gold,pred", [
        ("joyful", "joyful"),      # 答对
        ("sadness", "anger"),      # 答错
        ("neutral", "fear"),       # 答错（少数类）
        ("anger", "anger"),        # 答对（模型输出 == gold）
    ])
    def test_both_sides_isomorphic_short_format(self, gold, pred):
        """两侧同构：chosen/rejected 都是 `f" {label}"` 单标签短格式（S3）。"""
        builder = GoldLabelPairBuilder()
        pair = builder.build(make_outcome(pred), gold, prompt="p")
        for field in ("chosen", "rejected"):
            value = getattr(pair, field)
            assert value in SHORT_LABEL_FORMATS, (
                f"{field} 非同构短格式 \" {{label}}\": {value!r}"
            )
        assert pair.chosen != pair.rejected, (
            "偏好对两侧相同则无对比信号（gold 与错误答案必须可区分）"
        )

    def test_prompt_passed_through(self):
        pair = GoldLabelPairBuilder().build(
            make_outcome("joyful"), "joyful", prompt="prompt-xyz"
        )
        assert pair.prompt == "prompt-xyz"


class TestIgnoresConvergenceState:
    """b) 不读取辩论收敛状态字段（金标签判定与收敛状态正交，S3/V3 6.2 步骤 6）。"""

    @pytest.mark.parametrize("gold,pred", [("joyful", "joyful"), ("sadness", "anger")])
    def test_converged_true_false_produce_identical_pair(self, gold, pred):
        builder = GoldLabelPairBuilder()
        pair_true = builder.build(
            make_outcome(pred, converged=True, reask_triggered=True), gold, prompt="p"
        )
        pair_false = builder.build(
            make_outcome(pred, converged=False, reask_triggered=False), gold, prompt="p"
        )
        assert pair_true == pair_false, (
            "偏好对不应依赖 converged / reask_triggered（收敛状态只影响辩论，"
            "不影响金标签判定）"
        )

    def test_pair_value_only_depends_on_label_and_gold(self):
        builder = GoldLabelPairBuilder()
        # 收敛的答错（consensus but wrong）与未收敛的答错构造出同一偏好对
        wrong_converged = DebateOutcome(
            final_label="fear", confidence=0.95, rounds=3, converged=True,
            final_reasoning="agree on fear",
        )
        wrong_unconverged = DebateOutcome(
            final_label="fear", confidence=0.4, rounds=1, converged=False,
            final_reasoning="",
        )
        p1 = builder.build(wrong_converged, "surprise", prompt="p")
        p2 = builder.build(wrong_unconverged, "surprise", prompt="p")
        assert (p1.chosen, p1.rejected) == (p2.chosen, p2.rejected) == (
            " surprise", " fear",
        )


class TestReasoningNotInPair:
    """c) final_reasoning 不出现在 chosen/rejected 文本中（同构性红线）。"""

    def test_final_reasoning_excluded(self):
        reasoning = (
            "Because the speaker lost the game, he feels sadness. "
            "Label: sadness\nConfidence: 0.9"
        )
        pair = GoldLabelPairBuilder().build(
            make_outcome("sadness", final_reasoning=reasoning),
            "sadness",
            prompt="p",
        )
        for field in ("chosen", "rejected"):
            value = getattr(pair, field)
            assert reasoning not in value
            assert value in SHORT_LABEL_FORMATS, (
                f"{field} 混入了推理文本（须为 \" {{label}}\" 短格式）: {value!r}"
            )

    def test_reasoning_with_label_word_does_not_leak_into_wrong_side(self):
        """推理文本含错误标签词时，答错样本的 rejected 仍是模型标签而非推理片段。"""
        gold, model_label = "anger", "joyful"
        reasoning = f"the utterance shows joy not {gold}, Label: {model_label}"
        pair = GoldLabelPairBuilder().build(
            make_outcome(model_label, final_reasoning=reasoning), gold, prompt="p"
        )
        assert pair.chosen == f" {gold}"
        assert pair.rejected == f" {model_label}"
        assert len(pair.rejected) == len(model_label) + 1  # " {label}"，无任何附加文本

    def test_error_pattern_keeps_source_info(self):
        """错误模式信息只经 source_error_pattern 留作分组分析（不进偏好对文本）。"""
        pair = GoldLabelPairBuilder().build(
            make_outcome("disgust"), "anger", prompt="p"
        )
        assert pair.source_error_pattern == "disgust->anger"


class TestWrongLabelFallback:
    """混淆先验映射不到 / 等于 gold / 非法时，回退 schema 中第一个 ≠ gold 的标签。"""

    def test_empty_confusion_prior_falls_back_to_first_legal_label(self):
        builder = GoldLabelPairBuilder(confusion_prior={})
        pair = builder.build(make_outcome("neutral"), "neutral", prompt="p")
        # MELD_SCHEMA.labels[0] == "neutral" == gold → 回退下一个 ≠ gold 的标签
        fallback = next(lb for lb in MELD_SCHEMA.labels if lb != "neutral")
        assert pair.rejected == f" {fallback}"

    def test_confusion_prior_equal_to_gold_falls_back(self):
        # 先验恰好指向 gold 时不能用 gold 当 rejected（否则两侧相同）
        builder = GoldLabelPairBuilder(confusion_prior={"joyful": "joyful"})
        pair = builder.build(make_outcome("joyful"), "joyful", prompt="p")
        assert pair.rejected != pair.chosen
        assert pair.rejected in SHORT_LABEL_FORMATS


class TestBuildPairsBatch:
    """批量构造：与逐条构建一致；长度不一致时报错。"""

    def test_batch_matches_single(self):
        outcomes = [make_outcome("joyful"), make_outcome("anger"), make_outcome("fear")]
        golds = ["joyful", "sadness", "fear"]
        prompts = ["p1", "p2", "p3"]
        batch = build_pairs(outcomes, golds, prompts)
        singles = [
            GoldLabelPairBuilder().build(o, g, p)
            for o, g, p in zip(outcomes, golds, prompts)
        ]
        assert batch == singles

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="长度不一致"):
            build_pairs([make_outcome("joyful")], ["joyful", "fear"], ["p"])
