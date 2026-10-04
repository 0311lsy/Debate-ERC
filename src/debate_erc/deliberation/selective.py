"""选择性辩论（Selective Deliberation）——Layer 1 修复后的辩论决策机制。

审稿结论驱动的重设计（证据见 outputs/eval_greedy.json 与
eval_logprob_layer0.json）：
1. 旧辩论对所有样本无差别触发（53%），把大量高置信样本（该桶 acc 90.4%）
   拖入未训练格式的多轮噪声，净伤害 W-F1；
2. 旧置信度来自自报文本（DirectClassifyAgent 恒 0.5），门控全面失真；
3. 纯 SFT 底座的标签序列概率分桶校准单调（top1 概率 <0.5 桶 acc 0.48，
   ≥0.9 桶 acc 0.90），是可用的"困难度信号"。

机制（全部决策在概率空间，输出格式与 SFT 同构）：
- Proponent 视图：标准 classify 提示 → 贪心生成标签 y_a（主预测，与
  68.60 基线同口径）+ 标签序列后验 P_A，信念强度 p_a=P_A[y_a]；
- 概率门控：p_a ≥ τ 走高置信捷径，不触发辩论；
- Critic 视图（仅低置信样本）：同一 classify 提示注入固定的"三类 ERC
  典型错误自检" reasoning_hint（输出仍是 " {label}"，不引入因果链等
  未训练格式）→ 贪心生成 y_b + 后验 P_B；
- 采纳规则：critic 必须同时给出【不同标签】且【更强信念】
  （P_B[y_b] > P_A[y_a] + margin）才改判，否则保留 proponent 结论。
  标签以双方贪心生成为准（避免序列 logprob 长度偏差直接决定标签），
  概率只负责"是否信任翻案"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..agents.prompt_builder import build_classify_prompt
from ..utils.label_schema import LabelSchema
from ..utils.post_process import parse_label

if TYPE_CHECKING:
    from ..data.schemas import DialogueContext
    from ..models.backbone import LLMGenerator

# Critic 视角的固定自检指引：不依赖标签/样本、可在论文中完整陈述。
CRITIC_REASONING_HINT = (
    "Before answering, critically re-examine the target utterance for three "
    "common ERC errors:\n"
    "1. Trigger-emotion mismatch: an emotion word or punctuation alone does "
    "not determine the label; judge what the speaker actually feels "
    "(rhetorical questions, idioms, politeness formulas).\n"
    "2. Context reversal: contrastive connectives (but, actually, however), "
    "negation, or replies correcting the previous speaker may reverse the "
    "apparent emotion.\n"
    "3. Sarcasm and irony: literally positive words can convey negative "
    "emotion given the speaker dynamics and prior turns.\n"
    "If your first impression could fall into any of these errors, choose the "
    "label best supported by the full conversation instead."
)


@dataclass
class DeliberationResult:
    """单样本选择性辩论结果（供细粒度分析与误差归因）。"""

    final_label: str
    proponent_label: str
    proponent_prob: float
    triggered: bool
    flipped: bool
    critic_label: str | None = None
    critic_prob: float | None = None

    def to_dict(self) -> dict:
        return {
            "final_label": self.final_label,
            "proponent_label": self.proponent_label,
            "proponent_prob": round(self.proponent_prob, 4),
            "triggered": self.triggered,
            "flipped": self.flipped,
            "critic_label": self.critic_label,
            "critic_prob": None if self.critic_prob is None else round(self.critic_prob, 4),
        }


class SelectiveDeliberator:
    """概率门控的双视图辩论器（proponent 提出 / critic 质询 / 严格改判）。"""

    def __init__(
        self,
        generator: "LLMGenerator",
        schema: LabelSchema,
        gate_tau: float = 0.5,
        adopt_margin: float = 0.0,
        critic_hint: str | None = None,
        max_new_tokens: int = 8,
    ):
        self.generator = generator
        self.schema = schema
        self.gate_tau = float(gate_tau)
        self.adopt_margin = float(adopt_margin)
        # None=未指定则用默认自检指引；传空串/其他文本表示显式覆盖（hint 消融用）
        self.critic_hint = CRITIC_REASONING_HINT if critic_hint is None else critic_hint
        # 贪心短生成：classify 答案 SFT 目标即 " {label}\n"，8 token 足够
        self.gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        }

    def _greedy_label(self, prompt: str) -> str:
        """贪心生成并解析为合法标签（非法文本经 parse_label 映射进 schema）。"""
        raw = self.generator.generate(prompt, **self.gen_kwargs)
        return parse_label(raw, self.schema)

    def run_one(self, context: "DialogueContext") -> DeliberationResult:
        labels = list(self.schema.labels)

        # --- Proponent 视图（全部样本）---
        prompt_a = build_classify_prompt(context, self.schema)
        y_a = self._greedy_label(prompt_a)
        p_a_dist = self.generator.score_labels(prompt_a, labels)
        p_a = p_a_dist[y_a]

        # --- 概率门控：高置信捷径 ---
        if p_a >= self.gate_tau:
            return DeliberationResult(
                final_label=y_a,
                proponent_label=y_a,
                proponent_prob=p_a,
                triggered=False,
                flipped=False,
            )

        # --- Critic 视图（仅低置信样本，同构 classify 提示）---
        prompt_b = build_classify_prompt(
            context, self.schema, reasoning_hint=self.critic_hint
        )
        y_b = self._greedy_label(prompt_b)
        p_b_dist = self.generator.score_labels(prompt_b, labels)
        p_b = p_b_dist[y_b]

        # --- 严格采纳规则：标签不同 + critic 信念显著更强 ---
        flip = (y_b != y_a) and (p_b > p_a + self.adopt_margin)
        return DeliberationResult(
            final_label=y_b if flip else y_a,
            proponent_label=y_a,
            proponent_prob=p_a,
            triggered=True,
            flipped=flip,
            critic_label=y_b,
            critic_prob=p_b,
        )
