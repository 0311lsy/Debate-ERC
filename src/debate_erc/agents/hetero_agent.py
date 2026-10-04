"""异构模型 Agent（V3 六、消融 agent2=hetero / 设计报告 4.2）。

包装一个独立加载的异构模型（如 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B，
见 config.DebateConfig.hetero_model），按 role 复用提出方或批判方的
提示模板与解析逻辑（继承 BaseAgent，组合复用，不复制解析代码）。
"""

from __future__ import annotations

from ..data.schemas import AgentOutput, DialogueContext
from .base_agent import BaseAgent
from .causal_agent import CausalProponentAgent
from .critic_agent import CriticAgent

ROLES = ("proponent", "critic")


class HeteroAgent(BaseAgent):
    """异构模型包装 Agent：role 决定复用 causal_chain.j2 还是 critique.j2。"""

    def __init__(
        self,
        generator,
        schema,
        generation_kwargs: dict | None = None,
        role: str = "proponent",
    ):
        if role not in ROLES:
            raise ValueError(f"role 必须是 {ROLES} 之一，收到: {role!r}")
        super().__init__(generator, schema, generation_kwargs)
        self.role = role
        delegate_cls = CausalProponentAgent if role == "proponent" else CriticAgent
        # 委托对象与本体共享同一 generator/schema/采样参数（同一异构模型）
        self._delegate: BaseAgent = delegate_cls(generator, schema, generation_kwargs)

    def infer(
        self,
        context: DialogueContext,
        reasoning_hint: str | None = None,
        critique_points: list[str] | None = None,
        history: str | None = None,
        proposal: str = "",
    ) -> AgentOutput:
        """按 role 转发到对应的 Agent 实现（提出方 / 批判方行为完全一致）。"""
        if self.role == "proponent":
            return self._delegate.infer(
                context,
                reasoning_hint=reasoning_hint,
                critique_points=critique_points,
                history=history,
            )
        return self._delegate.infer(context, reasoning_hint=reasoning_hint, proposal=proposal)
