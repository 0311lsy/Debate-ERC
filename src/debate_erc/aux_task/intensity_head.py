"""辅助任务头（V3 6.8.4 / 设计报告 2.1）：样本流 → 辅助任务 DialogueSample 流。

两个辅助任务（均实现为"任务特定 adapter + 独立提示模板"）：
- IntensityEstimator：VADER 派生 strong/medium/weak 三档强度分类（V3 6.8.4）；
- SpeakerEstimator：掩码目标说话人后预测说话人（InstructERC speaker 辅助任务同款口径）。

本模块只做 样本→辅助样本 的转换（gold 替换 / 上下文改造 / prompt 构建），
训练由 multitask_scheduler 调度独立 SFTTrainer 完成，推理阶段不参与主任务。
"""

from __future__ import annotations

from ..agents.prompt_builder import render
from ..data.schemas import DialogueContext, DialogueSample
from ..utils.label_schema import LabelSchema
from ..data import intensity_annotator  # data 层强度标注（VADER 优先，词典回退）

INTENSITY_SCHEMA = LabelSchema(("strong", "medium", "weak"), "intensity")

# 掩码标记：说话人任务中替换目标说话人（提示中不可泄漏答案）
SPEAKER_MASK = "[MASK]"


def build_intensity_prompt(context: DialogueContext, schema: LabelSchema) -> str:
    """强度估计提示（intensity.j2）。签名与 build_classify_prompt 对齐（可注入 SFTTrainer）。"""
    return render(
        "intensity.j2",
        dialogue=context.render(),
        target_speaker=context.target_speaker,
        target_utterance=context.target_utterance,
        labels=list(schema.labels),
    )


def build_speaker_prompt(context: DialogueContext, schema: LabelSchema) -> str:
    """说话人归属提示（speaker.j2）；context 的目标说话人已被 SpeakerEstimator 掩码。"""
    return render(
        "speaker.j2",
        dialogue=context.render(),
        target_speaker=context.target_speaker,  # 掩码后的值（[MASK]）
        target_utterance=context.target_utterance,
        labels=list(schema.labels),
    )


class IntensityEstimator:
    """情绪强度估计辅助任务（V3 6.8.4）：三档 strong / medium / weak。"""

    schema = INTENSITY_SCHEMA
    prompt_builder = staticmethod(build_intensity_prompt)

    def to_aux_samples(self, samples: list[DialogueSample]) -> list[DialogueSample]:
        """主任务样本 → 强度辅助样本（gold = intensity_label(目标话语)，上下文不变）。"""
        aux: list[DialogueSample] = []
        for s in samples:
            level = intensity_annotator.intensity_label(s.context.target_utterance)
            aux.append(DialogueSample(
                context=s.context,
                gold_label=level,
                synthetic=False,
                split=s.split,
            ))
        return aux

    def label_distribution(self, aux_samples: list[DialogueSample]) -> dict[str, int]:
        """强度三档分布（验收 2.4-①：落盘检查无明显退化到单类）。"""
        from collections import Counter

        cnt = Counter(s.gold_label for s in aux_samples)
        return {label: cnt.get(label, 0) for label in INTENSITY_SCHEMA.labels}


class SpeakerEstimator:
    """说话人归属辅助任务：掩码目标说话人，预测其身份（InstructERC 同款对比）。

    schema 从训练集观察到的说话人动态构建（MELD 为 Friends 六主角）。
    """

    prompt_builder = staticmethod(build_speaker_prompt)

    def __init__(self, speakers: list[str] | None = None):
        if speakers:
            uniq = sorted(set(speakers))
            self.schema = LabelSchema(tuple(uniq), "speaker")
        else:
            self.schema = None  # to_aux_samples 时从样本推断

    def to_aux_samples(self, samples: list[DialogueSample]) -> list[DialogueSample]:
        """主任务样本 → 说话人辅助样本（目标说话人替换为 [MASK]，gold = 原说话人）。"""
        if self.schema is None:
            speakers = sorted({s.context.target_speaker for s in samples})
            self.schema = LabelSchema(tuple(speakers), "speaker")
        aux: list[DialogueSample] = []
        for s in samples:
            ctx = s.context
            gold = ctx.target_speaker
            if gold not in self.schema.labels:  # 训练后出现新说话人：跳过（保持 schema 稳定）
                continue
            masked_speakers = tuple(
                SPEAKER_MASK if i == ctx.target_idx else spk
                for i, spk in enumerate(ctx.speakers)
            )
            masked_ctx = DialogueContext(
                utterances=ctx.utterances,
                speakers=masked_speakers,
                target_idx=ctx.target_idx,
                history_window=ctx.history_window,
                dialogue_id=ctx.dialogue_id,
            )
            aux.append(DialogueSample(
                context=masked_ctx,
                gold_label=gold,
                synthetic=False,
                split=s.split,
            ))
        return aux
