"""LLMGenerator.score_labels 单元测试（Layer 1 置信度基础设施）。

覆盖 Layer 0 审稿发现的核心缺陷：
- 旧 label_confidence 取 " label" 首 token，而 LLaMA2 词表下该 token 恒为
  公共空格符（所有标签相同）→ 概率退化为 1/7、完全无区分度；
- 新 score_labels 在标签内容子词序列上 teacher-forcing 累加 logprob，
  必须能区分标签、归一为合法概率分布、argmax 与注入的 logit 偏好一致。

测试用 TinyLM + 可控伪 tokenizer（不依赖真实权重/GPU）。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from debate_erc.models.backbone import LLMGenerator

_SPACE = 60
_BOS = 1
_PAD = 0

# 伪词表：标签 " x" = [空格符] + 内容 token（joyful/sadness 模拟多子词）
_LABEL_ENCODE = {
    " neutral": [_SPACE, 20],
    " anger": [_SPACE, 23],
    " fear": [_SPACE, 24],
    " joyful": [_SPACE, 21, 22],
    " sadness": [_SPACE, 25, 26],
}
_LABELS = ["neutral", "anger", "fear", "joyful", "sadness"]


class TinyLM(nn.Module):
    """最小因果 LM：logits = head(emb(input_ids))，可注入 token 偏好。"""

    def __init__(self, vocab: int = 64, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class LabelScoringTokenizer:
    """模拟 LLaMA2 语义：' label' 首 token 恒为公共空格符。"""

    bos_token_id = _BOS
    pad_token_id = _PAD
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 max_length=None, return_tensors=None):
        ids = [10 + (i % 40) for i in range(len(text.split()))]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def encode(self, text, add_special_tokens=True):
        return list(_LABEL_ENCODE.get(text, []))

    def decode(self, ids, skip_special_tokens=True):
        return "x"


def make_generator(bias: dict[int, float] | None = None):
    model = TinyLM()
    if bias:
        with torch.no_grad():
            for tid, val in bias.items():
                model.head.bias[tid] = val
    tok = LabelScoringTokenizer()
    gen = LLMGenerator(model, tok)
    gen.candidate_labels = _LABELS
    return gen


class TestScoreLabels:
    def test_returns_valid_distribution(self):
        gen = make_generator()
        out = gen.score_labels("Answer:")
        assert set(out.keys()) == set(_LABELS)
        assert abs(sum(out.values()) - 1.0) < 1e-5
        assert all(0.0 <= p <= 1.0 for p in out.values())

    def test_distinguishes_labels_against_space_bug(self):
        """回归：公共空格首 token 不得让所有标签概率相等（旧 bug 退化 1/7）。"""
        gen = make_generator()
        out = gen.score_labels("Answer:")
        probs = list(out.values())
        assert max(probs) - min(probs) > 1e-6, "标签概率完全相同：首 token 退化 bug 复现"

    def test_argmax_follows_single_token_preference(self):
        """neutral 内容 token logit 被拉高 → argmax=neutral，且概率接近 1。"""
        gen = make_generator(bias={20: 50.0})
        out = gen.score_labels("Answer:")
        assert max(out, key=out.get) == "neutral"
        assert out["neutral"] > 0.99

    def test_multi_subword_label_can_win(self):
        """多子词 joyful 两个内容 token 都高 → 长度累加后压过短标签。"""
        gen = make_generator(bias={21: 40.0, 22: 40.0})
        out = gen.score_labels("Answer:")
        assert max(out, key=out.get) == "joyful"

    def test_probabilities_reflect_relative_strength(self):
        """两标签各占一个高 logit 内容 token 时，概率排序与偏好强度一致。"""
        gen = make_generator(bias={20: 5.0, 23: 2.0})
        out = gen.score_labels("Answer:")
        assert out["neutral"] > out["anger"]

    def test_legacy_label_confidence_forwards_to_scoring(self):
        """旧接口不再退化：转发 score_labels 后有区分度且 ∈ [0,1]。"""
        gen = make_generator(bias={24: 30.0})
        conf = gen.label_confidence("Answer:", "fear")
        assert 0.0 <= conf <= 1.0
        assert conf > 0.99

    def test_raises_without_labels(self):
        gen = make_generator()
        gen.candidate_labels = None
        try:
            gen.score_labels("Answer:")
        except ValueError:
            return
        raise AssertionError("无标签候选时应 raise ValueError")


class TestSingleTokenBPEVocab:
    """Qwen 式 BPE：' label' 是单个整词 token（无独立空格符），不得报错。"""

    def test_single_token_labels_scored(self, monkeypatch):
        tok = LabelScoringTokenizer()
        # 覆盖 encode：整词单 token，且 token id 互不相同
        mono = {
            " neutral": [40], " anger": [41], " fear": [42],
            " joyful": [43], " sadness": [44],
        }
        tok.encode = lambda text, add_special_tokens=True: list(mono.get(text, []))
        model = TinyLM()
        with torch.no_grad():
            model.head.bias[43] = 30.0  # joyful
        gen = LLMGenerator(model, tok)
        gen.candidate_labels = _LABELS
        out = gen.score_labels("Answer:")
        assert abs(sum(out.values()) - 1.0) < 1e-5
        assert max(out, key=out.get) == "joyful"
