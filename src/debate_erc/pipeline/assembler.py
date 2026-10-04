"""PipelineAssembler（设计报告 5.2 / 7.2）：配置 → 运行时对象 → Stage 列表。

装配规则：
- 全部组件由 ExperimentConfig 布尔开关驱动，组合空间 = 开关向量空间；
- 依赖约束校验在 config.validate_config 完成（reask 依赖 debate、dpo 需要偏好对来源），
  本模块只按开关构造对象，遇到非法组合（如 debate 关闭却要求构造辩论循环）直接报错；
- 异构辩论（agent2=hetero）需要额外加载异构模型，由调用方传入 hetero_generator，
  本模块不做模型加载（保持装配纯函数性）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents import (
    BaseAgent,
    CausalProponentAgent,
    CriticAgent,
    DirectClassifyAgent,
    HeteroAgent,
)
from ..config import ExperimentConfig
from ..debate import ConvergenceChecker, DebateLoop, ReaskFallback
from ..reasoning import TemplateRouter
from ..routing import DifficultyRouter
from ..utils.label_schema import LabelSchema, get_schema
from .stages import (
    DebateStage,
    LabelStage,
    PipelineStage,
    ReasoningStage,
    RouteStage,
)


@dataclass
class Runtime:
    """一次实验的推理运行时（Agent / 辩论循环 / 路由器等共享对象）。"""

    schema: LabelSchema
    generator: object                      # LLMGenerator（主模型）
    proponent: BaseAgent
    critic: BaseAgent
    direct_agent: BaseAgent
    debate_loop: Optional[DebateLoop]
    template_router: TemplateRouter
    difficulty_router: DifficultyRouter
    generation_kwargs: dict = field(default_factory=dict)


def build_generation_kwargs(config: ExperimentConfig) -> dict:
    """构造 LLMGenerator.generate 的推理参数（dev 偏好对生成与 test 评估共用）。

    Layer 0 修复：推理口径取 config.generation.eval_do_sample（默认 False，
    贪心解码），保证所有数字可复现；贪心时 temperature/top_p 显式置 1.0，
    避免 HF 采样参数告警。
    """
    eval_sample = config.generation.eval_do_sample
    return {
        "max_new_tokens": config.generation.max_new_tokens,
        "temperature": config.generation.temperature if eval_sample else 1.0,
        "top_p": config.generation.top_p if eval_sample else 1.0,
        "do_sample": eval_sample,
    }


def build_runtime(
    config: ExperimentConfig,
    generator,
    hetero_generator=None,
) -> Runtime:
    """按配置构建推理运行时。

    hetero_generator：agent2=hetero 时的异构模型生成后端（None 则退化为同款）。
    """
    schema = get_schema(config.data.dataset)
    gen_kwargs = build_generation_kwargs(config)

    proponent = CausalProponentAgent(generator, schema, gen_kwargs)
    if config.debate.agent2 == "hetero" and hetero_generator is not None:
        critic: BaseAgent = HeteroAgent(hetero_generator, schema, gen_kwargs, role="critic")
    else:
        critic = CriticAgent(generator, schema, gen_kwargs)
    direct_agent = DirectClassifyAgent(generator, schema, gen_kwargs)

    debate_loop: DebateLoop | None = None
    if config.debate.enabled:
        reask: ReaskFallback | None = None
        if config.reask.enabled:
            reask = ReaskFallback(
                low_confidence=config.reask.low_confidence,
                max_reask=config.reask.max_reask,
            )
        debate_loop = DebateLoop(
            proponent=proponent,
            critic=critic,
            convergence=ConvergenceChecker(config.debate.confidence_threshold),
            max_rounds=config.debate.max_rounds,
            reask=reask,
        )

    return Runtime(
        schema=schema,
        generator=generator,
        proponent=proponent,
        critic=critic,
        direct_agent=direct_agent,
        debate_loop=debate_loop,
        template_router=TemplateRouter(
            schema, long_utterance_chars=config.routing.long_threshold
        ),
        difficulty_router=DifficultyRouter(
            short_threshold=config.routing.short_threshold,
            long_threshold=config.routing.long_threshold,
            minority_boost=config.routing.minority_boost,
        ),
        generation_kwargs=gen_kwargs,
    )


class PipelineAssembler:
    """配置 → 按序启用的 Stage 列表（消融实验零代码改动的机制核心）。"""

    @staticmethod
    def assemble(config: ExperimentConfig, runtime: Runtime) -> list[PipelineStage]:
        stages: list[PipelineStage] = [
            RouteStage(config.routing.enabled, runtime.difficulty_router),
            ReasoningStage(config.reasoning.enabled, runtime.template_router),
            DebateStage(runtime.debate_loop, runtime.direct_agent),
            LabelStage(runtime.schema),
        ]
        # 装配级断言（config.validate_config 之后的第二道防线）
        if config.debate.enabled and runtime.debate_loop is None:
            raise ValueError("debate.enabled=true 但 debate_loop 未构建（Runtime 不完整）")
        return stages
