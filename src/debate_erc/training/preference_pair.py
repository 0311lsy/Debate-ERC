"""DPO 偏好对构建（V3 6.2 步骤 6，金标签判定；不基于收敛状态）。

PairSource 抽象：任何能从辩论结果 + 金标签构造偏好对的来源；
GoldLabelPairBuilder：金标签判定的默认实现（S3 契约：chosen/rejected 统一为
`f" {label}"` 同构短格式，与 SFT 答案 token 契约对齐）——
- 模型答对：chosen = " {final_label}"，rejected = 按混淆先验构造的错误答案 " {wrong}"；
- 模型答错：chosen = " {gold}"，rejected = " {final_label}"（模型自己的错误输出）；
- 推理文本（final_reasoning）不进入偏好对：长推理链与短答案混排会破坏
  chosen/rejected 的同构性（DPO 的对比信号被格式差异淹没）；
  错误模式信息只经 source_error_pattern 留作分组分析。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.schemas import DebateOutcome, PreferencePair
from ..utils.label_schema import MELD_SCHEMA, LabelSchema

# 混淆先验：某标签最常被误判成的类（V3 错误分析口径的默认值）
DEFAULT_CONFUSION_PRIOR: dict[str, str] = {
    "joyful": "neutral",
    "sadness": "neutral",
    "anger": "joyful",
    "neutral": "sadness",
    "surprise": "neutral",
    "disgust": "anger",
    "fear": "surprise",
}


class PairSource(ABC):
    """偏好对来源抽象：辩论结果 + 金标签 + prompt → PreferencePair。"""

    @abstractmethod
    def build(self, outcome: DebateOutcome, gold: str, prompt: str) -> PreferencePair:
        raise NotImplementedError


class GoldLabelPairBuilder(PairSource):
    """金标签判定（V3 6.2 步骤 6）：final_label 与 gold 的对错决定 chosen/rejected。"""

    def __init__(
        self,
        schema: LabelSchema = MELD_SCHEMA,
        confusion_prior: dict[str, str] | None = None,
    ):
        self.schema = schema
        self.confusion_prior: dict[str, str] = (
            dict(confusion_prior) if confusion_prior is not None
            else dict(DEFAULT_CONFUSION_PRIOR)
        )

    def _wrong_label(self, final_label: str, gold: str) -> str:
        """取混淆先验中 final_label 的最常见误判类；映射不到/非法/等于 gold 时，
        用 schema 里第一个 ≠ gold 的标签兜底。"""
        cand = self.confusion_prior.get(final_label)
        if cand is not None and cand != gold and self.schema.is_legal(cand):
            return cand
        for lb in self.schema.labels:
            if lb != gold:
                return lb
        raise ValueError(f"schema 只有一个标签，无法构造错误答案: {self.schema.labels}")

    def build(self, outcome: DebateOutcome, gold: str, prompt: str) -> PreferencePair:
        """单条构造（S3：chosen/rejected 同构短格式 " {label}"）。

        - outcome.final_label == gold（答对）：chosen = " {final_label}"，
          rejected = 混淆先验错误答案 " {wrong}"；
        - 答错：chosen = " {gold}"，rejected = " {final_label}"（模型自己的错误输出）；
        - final_reasoning 不进入偏好对（保持 chosen/rejected 同构），
          错误模式只经 source_error_pattern = "{final_label}->{gold}" 留作分组分析。
        """
        if outcome.final_label == gold:
            chosen = f" {outcome.final_label}"
            rejected = f" {self._wrong_label(outcome.final_label, gold)}"
        else:
            chosen = f" {gold}"
            rejected = f" {outcome.final_label}"
        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source_error_pattern=f"{outcome.final_label}->{gold}",
        )


def build_pairs(
    outcomes: list[DebateOutcome],
    golds: list[str],
    prompts: list[str],
    schema: LabelSchema = MELD_SCHEMA,
    confusion_prior: dict[str, str] | None = None,
) -> list[PreferencePair]:
    """批量构造偏好对（outcomes / golds / prompts 等长 zip）。"""
    if not (len(outcomes) == len(golds) == len(prompts)):
        raise ValueError(
            f"长度不一致: outcomes={len(outcomes)} golds={len(golds)} prompts={len(prompts)}"
        )
    builder = GoldLabelPairBuilder(schema=schema, confusion_prior=confusion_prior)
    return [
        builder.build(o, g, p) for o, g, p in zip(outcomes, golds, prompts)
    ]
