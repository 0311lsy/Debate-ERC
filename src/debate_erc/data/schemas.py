"""核心数据结构（设计报告 4.3）：DialogueContext / DialogueSample / AgentOutput 等。

放于 data 层供所有上层模块共享（依赖方向：上层 import 本模块）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DialogueContext:
    """一段对话中目标话语的上下文。

    utterances: 上下文话语列表（含目标话语，目标为最后一个元素之前的 target_idx 标记）
    speakers: 与 utterances 等长的说话人列表
    target_idx: 目标话语在 utterances 中的下标
    history_window: 构建上下文时使用的前序窗口大小
    """

    utterances: tuple[str, ...]
    speakers: tuple[str, ...]
    target_idx: int
    history_window: int
    dialogue_id: str = ""

    @property
    def target_utterance(self) -> str:
        return self.utterances[self.target_idx]

    @property
    def target_speaker(self) -> str:
        return self.speakers[self.target_idx]

    @property
    def history(self) -> tuple[str, ...]:
        """目标话语之前的话语（不含目标）。"""
        return self.utterances[: self.target_idx]

    def render(self, max_chars: int = 800) -> str:
        """渲染为对话文本（Speaker: utterance 每行一条），供提示词使用。

        超限时以目标行为锚点截断（R2-M3 修复）：目标行永不丢失（即使其本身
        超预算也完整保留），其余行从目标行向两侧就近扩展、放不下即停止。
        目标行位于末位时（当前全部生产构造点如此）等价于旧的保尾截断。
        """
        lines = []
        for i, (spk, utt) in enumerate(zip(self.speakers, self.utterances)):
            mark = " [TARGET]" if i == self.target_idx else ""
            lines.append(f"{spk}: {utt}{mark}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        kept_idx = {self.target_idx}
        used = len(lines[self.target_idx])
        lo, hi = self.target_idx - 1, self.target_idx + 1
        while lo >= 0 or hi < len(lines):
            added = False
            if lo >= 0:  # 每次加入紧邻行恰好新增 1 个换行分隔符
                cost = len(lines[lo]) + 1
                if used + cost <= max_chars:
                    kept_idx.add(lo)
                    used += cost
                    lo -= 1
                    added = True
            if hi < len(lines):
                cost = len(lines[hi]) + 1
                if used + cost <= max_chars:
                    kept_idx.add(hi)
                    used += cost
                    hi += 1
                    added = True
            if not added:
                break  # 两侧都放不下：停止（目标行已锁定）
        return "\n".join(lines[i] for i in sorted(kept_idx))


@dataclass(frozen=True)
class DialogueSample:
    """一条训练/评估样本。"""

    context: DialogueContext
    gold_label: str
    synthetic: bool = False  # 合成样本标记（红线 #6：真实子集单独汇报）
    split: str = "train"


@dataclass
class AgentOutput:
    """单个 Agent 的推断输出。"""

    label: str
    confidence: float
    causal_chain: str = ""          # Agent1: 触发事件 → 心理状态变化 → 情感标签
    critique_points: list[str] = field(default_factory=list)  # Agent2: 三类缺陷审查
    raw_text: str = ""


@dataclass
class DebateOutcome:
    """辩论全流程结果（设计报告 4.3）。"""

    final_label: str
    confidence: float
    rounds: int = 0
    converged: bool = False
    consensus_but_wrong: bool = False     # 错误共识（评估阶段对照金标签填充）
    reask_triggered: bool = False
    reask_rounds: int = 0
    proponent_output: AgentOutput | None = None
    critic_output: AgentOutput | None = None
    disagreement_points: list[str] = field(default_factory=list)
    final_reasoning: str = ""
    latency_seconds: float = 0.0
    first_round_label: str | None = None   # 辩论首轮 Agent1 标签（单 Agent 基线口径）
    reask_pre_label: str | None = None      # reask 前的 final_label（兜底修正率口径）
    generation_calls: int = 0               # 本样本 LLM 生成调用次数（每次 infer +1）


@dataclass(frozen=True)
class PreferencePair:
    """DPO 偏好对（金标签判定，V3 6.2 步骤 6）。"""

    prompt: str
    chosen: str
    rejected: str
    source_error_pattern: str = "generic"


@dataclass(frozen=True)
class MetricReport:
    """评估结果汇总。"""

    wf1: float
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_pr: dict[str, dict[str, float]]
