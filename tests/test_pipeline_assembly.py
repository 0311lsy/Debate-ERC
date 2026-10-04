"""Pipeline 装配与 Stage 流转测试（mock 生成器，不加载真实模型）。

覆盖：
1. base 配置（无辩论）→ 单 Agent 直接推理路径 + 推理链注入；
2. 辩论配置 → DebateLoop 路径（收敛 / 兜底触发）；
3. 路由配置 → 三模式分派（PERCEIVE 不加链、DELIBERATE 走辩论）；
4. 开关消融：关闭组件后 Stage 变 no-op，InferenceState 原样传递；
5. run_id 与 assembler 断言。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debate_erc.config import (
    DebateConfig,
    ExperimentConfig,
    ReasoningConfig,
    ReaskConfig,
    RoutingConfig,
)
from debate_erc.data.schemas import DialogueContext, DialogueSample
from debate_erc.pipeline import (
    PipelineAssembler,
    build_runtime,
    run_pipeline,
)
from debate_erc.routing import RouteMode
from debate_erc.utils.label_schema import MELD_SCHEMA

PROPONENT_REPLY = (
    "Trigger: a scary event\nState change: sudden alarm\nLabel: fear\nConfidence: 0.9"
)
CRITIC_REPLY = "Critique:\n- none\nLabel: fear\nConfidence: 0.8"
DIRECT_REPLY = "fear"


class MockGenerator:
    """按提示内容分派固定输出的 mock 生成后端。"""

    def __init__(self, direct_reply=DIRECT_REPLY):
        self.direct_reply = direct_reply
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if "Agent 2" in prompt:  # critique.j2 开头标识批判方
            return CRITIC_REPLY
        if "causal proponent" in prompt:  # causal_chain.j2 开头标识提出方
            return PROPONENT_REPLY
        return self.direct_reply  # classify.j2（单 Agent 直接推理）


def make_sample(target="Oh my god, look out! There is something moving behind you!",
                gold="fear", speakers=("Joey", "Chandler", "Joey")):
    ctx = DialogueContext(
        utterances=("Hi there.", "You have got to see this.", target),
        speakers=speakers,
        target_idx=2,
        history_window=2,
        dialogue_id="test-dlg",
    )
    return DialogueSample(context=ctx, gold_label=gold)


def make_config(**overrides) -> ExperimentConfig:
    base = ExperimentConfig()
    fields = {
        "debate": DebateConfig(enabled=overrides.pop("debate", False)),
        "reasoning": ReasoningConfig(enabled=overrides.pop("reasoning", False)),
        "routing": RoutingConfig(enabled=overrides.pop("routing", False)),
    }
    if "reask" in overrides:
        fields["reask"] = ReaskConfig(enabled=overrides.pop("reask"))
    return replace(base, **fields, **overrides)


def test_base_config_direct_path_with_hint():
    config = make_config(reasoning=True)
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    assert state.final_label == "fear"
    assert state.single_output is not None
    assert state.debate_outcome is None
    assert state.reasoning_hint is not None  # CoT 注入


def test_base_config_no_reasoning_no_hint():
    config = make_config()
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    assert state.final_label == "fear"
    assert state.reasoning_hint is None  # 无链基线


def test_debate_config_converges():
    config = make_config(debate=True, reasoning=True, reask=True)
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    assert state.debate_outcome is not None
    assert state.debate_outcome.converged  # 双方都是 fear 且置信度 ≥ 0.6
    assert state.debate_outcome.rounds == 1
    assert not state.debate_outcome.reask_triggered  # 收敛且高置信 → 不触发兜底
    assert state.final_label == "fear"


def test_debate_reask_triggered_on_low_confidence():
    low_prop = "Trigger: t\nState change: s\nLabel: fear\nConfidence: 0.3"
    low_crit = "Critique:\n- none\nLabel: fear\nConfidence: 0.3"

    class LowConfGen(MockGenerator):
        def generate(self, prompt, **kw):
            if "Agent 2" in prompt:
                return low_crit
            if "causal proponent" in prompt:
                return low_prop
            return self.direct_reply

    config = make_config(debate=True, reask=True)
    runtime = build_runtime(config, LowConfGen())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    # 标签一致但置信度 < 0.6 → ConvergenceChecker 判不收敛 → 触发兜底
    assert not state.debate_outcome.converged
    assert state.debate_outcome.reask_triggered
    assert state.final_label in MELD_SCHEMA.labels  # 兜底后标签仍合法


def test_debate_reask_condition2_converged_but_low_confidence():
    """R2-m16：reask 条件 2（收敛但低置信）集成路径。

    置信度落在 [confidence_threshold=0.6, low_confidence=0.75) 区间
    （0.7 / 0.65 → 均值 0.675）→ 辩论收敛但触发兜底，且走 reask.j2
    空分歧点分支（critic 无审查点）。
    """
    mid_prop = "Trigger: t\nState change: s\nLabel: fear\nConfidence: 0.7"
    mid_crit = "Critique:\n- none\nLabel: fear\nConfidence: 0.65"

    class MidConfGen(MockGenerator):
        def generate(self, prompt, **kw):
            if "Agent 2" in prompt:
                return mid_crit
            if "causal proponent" in prompt:
                return mid_prop
            return self.direct_reply

    config = make_config(debate=True, reask=True)
    runtime = build_runtime(config, MidConfGen())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    outcome = state.debate_outcome
    assert outcome is not None
    assert outcome.converged  # 标签一致且置信度 ≥ 0.6 → 条件 1 不触发
    assert outcome.reask_triggered  # 均值 0.675 < 0.75 → 条件 2 触发
    assert outcome.reask_pre_label == "fear"  # 兜底前的结论被记录
    assert outcome.reask_rounds == 1
    assert outcome.generation_calls == 3  # 辩论 2 次 + reask 1 次
    assert state.final_label in MELD_SCHEMA.labels


def test_routing_perceive_skips_hint_and_debate():
    config = make_config(debate=True, reasoning=True, routing=True)
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    # 短话语 + 无说话人切换 + 上下文 ≤ 2 轮 → PERCEIVE
    sample = DialogueSample(
        context=DialogueContext(
            utterances=("Fine.",), speakers=("A",),
            target_idx=0, history_window=1, dialogue_id="short",
        ),
        gold_label="neutral",
    )
    state = run_pipeline(stages, sample)
    assert state.mode == RouteMode.PERCEIVE
    assert state.reasoning_hint is None  # 感知模式不加链
    assert state.single_output is not None  # 走单 Agent 轻量路径
    assert state.debate_outcome is None


def test_routing_deliberate_uses_debate():
    config = make_config(debate=True, reasoning=True, routing=True)
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    # 长话语（≥150 字符）→ DELIBERATE
    long_target = (
        "I cannot believe this is happening to me, honestly, of all the days "
        "it had to be today, right before the big presentation, and now I am "
        "standing here without any of the slides I spent the whole week making."
    )
    state = run_pipeline(stages, make_sample(target=long_target))
    assert state.mode == RouteMode.DELIBERATE
    assert state.debate_outcome is not None


def test_stage_disabled_is_noop():
    # reasoning 关闭：ReasoningStage.apply 原样传递
    config = make_config(reasoning=False)
    runtime = build_runtime(config, MockGenerator())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    assert state.reasoning_hint is None


def test_label_stage_always_legal():
    class GarbageGen(MockGenerator):
        def generate(self, prompt, **kw):
            if "Agent 2" in prompt or "causal proponent" in prompt:
                return "Label: angr\nConfidence: 0.9"  # 非法标签
            return "some garbage output without a label"

    config = make_config()
    runtime = build_runtime(config, GarbageGen())
    stages = PipelineAssembler.assemble(config, runtime)
    state = run_pipeline(stages, make_sample())
    assert state.final_label in MELD_SCHEMA.labels  # post_process 合法化兜底
