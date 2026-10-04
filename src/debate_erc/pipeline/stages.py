"""Pipeline Stage（设计报告 5.2）：消融开关的载体。

每个创新组件实现为一个可插拔 Stage，对 InferenceState 依次变换：
  RouteStage → ReasoningStage → DebateStage → LabelStage

- 关闭某开关 → 对应 Stage 成为 no-op，InferenceState 原样传递（消融零代码改动）；
- 阶段顺序说明：设计报告 5.2 列举的顺序为 Reasoning→Debate→Reask→Route→Label，
  本实现把 Route 前置（路由决定该样本走辩论还是轻量路径，语义上必须先判），
  Reask 并入 DebateLoop 内部（DebateLoop.run 已集成 ReaskFallback，与 V3 6.7.3 一致）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..agents.prompt_builder import build_classify_prompt
from ..data.schemas import AgentOutput, DebateOutcome, DialogueSample
from ..routing.difficulty_router import RouteMode
from ..utils.label_schema import LabelSchema, map_illegal_label

if TYPE_CHECKING:
    from ..agents import BaseAgent
    from ..debate import DebateLoop
    from ..reasoning import TemplateRouter
    from ..routing import DifficultyRouter


@dataclass
class InferenceState:
    """单样本在流水线中的可变状态（各 Stage 就地读写）。"""

    sample: DialogueSample
    mode: RouteMode | None = None        # 路由结果（routing 关闭时为 None）
    forced_mode: RouteMode | None = None  # Oracle 模式下强制指定（三模式分别跑）
    reasoning_hint: str | None = None    # ReasoningStage 注入的推理指引
    single_output: AgentOutput | None = None   # 单 Agent 路径输出
    debate_outcome: DebateOutcome | None = None  # 辩论路径输出
    prompt: str = ""                     # 基础分类提示（偏好对构造 / 审计用）
    final_label: str | None = None       # LabelStage 产出


class PipelineStage(ABC):
    """流水线阶段基类：enabled 由配置注入，关闭时 apply 为 no-op。"""

    name: str = "stage"

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def apply(self, state: InferenceState) -> InferenceState:
        """enabled=False 时原样传递（设计报告 5.2 的消融机制核心）。"""
        if self.enabled:
            return self._apply(state)
        return state

    @abstractmethod
    def _apply(self, state: InferenceState) -> InferenceState:
        raise NotImplementedError


class RouteStage(PipelineStage):
    """难度路由（V3 7.2）：PERCEIVE / COMPOSE / DELIBERATE 三模式。"""

    name = "route"

    def __init__(self, enabled: bool, router: DifficultyRouter):
        super().__init__(enabled)
        self.router = router

    def _apply(self, state: InferenceState) -> InferenceState:
        state.mode = state.forced_mode or self.router.route(state.sample.context)
        return state


class ReasoningStage(PipelineStage):
    """推理链注入（V3 7.1）：按错误模式选模板，写入 reasoning_hint。

    PERCEIVE（轻量感知）模式不加链；routing 关闭（mode=None）时始终注入
    （reason_erc / debate_full 的默认行为）。
    """

    name = "reasoning"

    def __init__(self, enabled: bool, template_router: TemplateRouter):
        super().__init__(enabled)
        self.template_router = template_router

    def _apply(self, state: InferenceState) -> InferenceState:
        if state.mode is RouteMode.PERCEIVE:
            return state
        state.reasoning_hint = self.template_router.build_hint(state.sample.context)
        return state


class DebateStage(PipelineStage):
    """辩论路径与单 Agent 路径的分派（恒启用：推理必须产出结果）。

    - debate_loop 存在（debate.enabled=true）且 mode 为 DELIBERATE
      （或路由关闭 mode=None）→ DebateLoop 完整辩论；
    - 其余情况（debate 关闭，或路由判定轻量）→ DirectClassifyAgent 直接分类。
    本 Stage 不做消融开关：辩论的消融 = 走单 Agent 路径（V3 单 Agent+CoT 基线）。
    """

    name = "debate"

    def __init__(
        self,
        debate_loop: DebateLoop | None,
        direct_agent: BaseAgent,
    ):
        super().__init__(True)
        self.debate_loop = debate_loop
        self.direct_agent = direct_agent

    def _apply(self, state: InferenceState) -> InferenceState:
        if self.debate_loop is not None and state.mode in (None, RouteMode.DELIBERATE):
            state.debate_outcome = self.debate_loop.run(
                state.sample.context, reasoning_hint=state.reasoning_hint
            )
        else:
            state.single_output = self.direct_agent.infer(
                state.sample.context, reasoning_hint=state.reasoning_hint
            )
        return state


class LabelStage(PipelineStage):
    """终标签确定与合法化：辩论结果优先，单 Agent 输出兜底。

    始终启用（不可消融）；同时落基础分类 prompt（偏好对构造 / 审计）。
    """

    name = "label"

    def __init__(self, schema: LabelSchema):
        super().__init__(True)
        self.schema = schema

    def _apply(self, state: InferenceState) -> InferenceState:
        if state.debate_outcome is not None:
            state.final_label = state.debate_outcome.final_label
        elif state.single_output is not None:
            state.final_label = state.single_output.label
        if state.final_label is None:
            raise RuntimeError("LabelStage 前无任何推断输出（DebateStage 未执行？）")
        state.final_label = map_illegal_label(state.final_label, self.schema)
        state.prompt = build_classify_prompt(state.sample.context, self.schema)
        return state


def run_pipeline(stages: list[PipelineStage], sample: DialogueSample,
                 forced_mode: RouteMode | None = None) -> InferenceState:
    """按序执行全部 Stage（Oracle 模式用 forced_mode 强制路由结果）。"""
    state = InferenceState(sample=sample, forced_mode=forced_mode)
    for stage in stages:
        stage.apply(state)
    return state
