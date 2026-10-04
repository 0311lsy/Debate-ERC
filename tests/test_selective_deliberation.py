"""SelectiveDeliberator 机制测试（mock 生成器，不依赖真实权重）。

覆盖审稿要求的四条决策路径：
1. 高置信 → 捷径不触发；
2. 低置信 + critic 同标签 → 不翻案；
3. 低置信 + critic 异标签且信念更强（超 margin）→ 翻案；
4. 低置信 + critic 异标签但信念不足 → 保守不翻案。
"""

from __future__ import annotations

from debate_erc.data.schemas import DialogueContext
from debate_erc.deliberation import SelectiveDeliberator
from debate_erc.utils.label_schema import MELD_SCHEMA

_LABELS = list(MELD_SCHEMA.labels)


def make_context():
    return DialogueContext(
        utterances=("hello there", "I am furious"),
        speakers=("A", "B"),
        target_idx=1,
        history_window=5,
        dialogue_id="d0",
    )


class FakeGenerator:
    """按 prompt 类型（默认/critic hint）返回预设生成标签与概率分布。"""

    def __init__(self, gen_a: str, dist_a: dict, gen_b: str | None = None,
                 dist_b: dict | None = None):
        self.gen_a, self.dist_a = gen_a, dist_a
        self.gen_b, self.dist_b = gen_b, dist_b

    def generate(self, prompt, **kw):
        if self.gen_b is not None and "common ERC errors" in prompt:
            return f" {self.gen_b}\n"
        return f" {self.gen_a}\n"

    def score_labels(self, prompt, labels=None, max_length=2048):
        if self.dist_b is not None and "common ERC errors" in prompt:
            return dict(self.dist_b)
        return dict(self.dist_a)


def dist(top: str, p: float) -> dict:
    """生成以 top 为主、其余均分剩余概率的分布。"""
    rest = (1.0 - p) / (len(_LABELS) - 1)
    return {lb: (p if lb == top else rest) for lb in _LABELS}


class TestSelectiveDeliberator:
    def test_high_confidence_short_circuits(self):
        gen = FakeGenerator("neutral", dist("neutral", 0.95))
        d = SelectiveDeliberator(gen, MELD_SCHEMA, gate_tau=0.5)
        r = d.run_one(make_context())
        assert r.triggered is False and r.flipped is False
        assert r.final_label == "neutral"
        assert r.critic_label is None  # critic 视图未调用

    def test_low_confidence_agree_no_flip(self):
        gen = FakeGenerator("neutral", dist("neutral", 0.30),
                            "neutral", dist("neutral", 0.35))
        d = SelectiveDeliberator(gen, MELD_SCHEMA, gate_tau=0.5, adopt_margin=0.0)
        r = d.run_one(make_context())
        assert r.triggered is True and r.flipped is False
        assert r.final_label == "neutral"

    def test_low_confidence_critic_stronger_flips(self):
        gen = FakeGenerator("neutral", dist("neutral", 0.30),
                            "anger", dist("anger", 0.80))
        d = SelectiveDeliberator(gen, MELD_SCHEMA, gate_tau=0.5, adopt_margin=0.0)
        r = d.run_one(make_context())
        assert r.triggered is True and r.flipped is True
        assert r.final_label == "anger"

    def test_critic_stronger_but_below_margin_no_flip(self):
        gen = FakeGenerator("neutral", dist("neutral", 0.40),
                            "anger", dist("anger", 0.42))
        d = SelectiveDeliberator(gen, MELD_SCHEMA, gate_tau=0.5, adopt_margin=0.1)
        r = d.run_one(make_context())
        assert r.triggered is True and r.flipped is False
        assert r.final_label == "neutral"

    def test_boundary_probability_triggers_gate(self):
        """p_a 恰好等于 τ 时不触发（捷径为闭区间）。"""
        gen = FakeGenerator("neutral", dist("neutral", 0.50),
                            "anger", dist("anger", 0.90))
        d = SelectiveDeliberator(gen, MELD_SCHEMA, gate_tau=0.5)
        r = d.run_one(make_context())
        assert r.triggered is False and r.final_label == "neutral"
