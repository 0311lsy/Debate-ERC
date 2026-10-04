"""训练模块回归测试（R2-M2）：锁定第一轮修复的核心数值路径。

覆盖（全部不加载真实模型，tiny 模块 + 伪 tokenizer）：
1. left_truncate_with_bos：未超长原样 / 保尾 + BOS 重前置 / 无 BOS 口径；
2. SFTDataset：prompt 不做 tokenizer 右截断（R2-m4），超长保护全权由
   left_truncate_with_bos 承担（尾部目标话语保留）；
3. SFTTrainer.train：单 epoch 步数 / eval 前后模式切换 / steps_budget 截断
   跳过 epoch 收尾（不落盘 checkpoint、不计入 epochs_done）/ warmup 线性调度；
4. DPOTrainer：非 PeftModel 拒绝（S3 契约）/ 训练 + adapter-only 落盘 /
   max_steps 截断跳过 epoch 落盘（R2-m6）；
5. MultitaskScheduler：weight=0 → 辅助任务零步（R2-m5）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from debate_erc.aux_task.multitask_scheduler import MultitaskScheduler  # noqa: E402
from debate_erc.config import TrainingConfig  # noqa: E402
from debate_erc.data.schemas import (  # noqa: E402
    DialogueContext,
    DialogueSample,
    PreferencePair,
)
from debate_erc.training.dpo_trainer import DPOTrainer  # noqa: E402
from debate_erc.training.sft_trainer import (  # noqa: E402
    SFTTrainer,
    _SFTDataset,
    left_truncate_with_bos,
)
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402

_BOS = 1


class TinyLM(nn.Module):
    """最小因果 LM 替身：input_ids → logits [B, T, V]。"""

    def __init__(self, vocab: int = 64, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class FakeTokenizer:
    """伪 tokenizer：按空白切词编码（模拟 HF 语义：truncation=True 时右截断）。"""

    bos_token_id = _BOS
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [10 + (i % 50) for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids):
        return "label" if ids else ""


def make_samples(n: int = 2) -> list[DialogueSample]:
    return [
        DialogueSample(
            context=DialogueContext(
                utterances=("hello there", "I am fine today"),
                speakers=("A", "B"),
                target_idx=1,
                history_window=1,
                dialogue_id=f"d{i}",
            ),
            gold_label="anger",
        )
        for i in range(n)
    ]


def make_pairs(n: int = 2) -> list[PreferencePair]:
    return [
        PreferencePair(prompt=f"prompt {i}", chosen=" anger", rejected=" joy")
        for i in range(n)
    ]


class DummyAdapterManager:
    """MultitaskScheduler 仅需 activate()（训练器内部不依赖 PEFT）。"""

    active = None

    def activate(self, name):
        self.active_name = name


# ---------------------------------------------------------------------------
# 1. left_truncate_with_bos
# ---------------------------------------------------------------------------

class TestLeftTruncateWithBos:
    def test_noop_when_under_limit(self):
        ids = [5, 6, 7]
        out, total = left_truncate_with_bos(ids, 2, _BOS, 100)
        assert out is ids  # 未超长：原对象返回
        assert total == 5

    def test_keeps_tail_and_prepends_bos(self):
        ids = list(range(2, 102))  # 100 个 prompt token
        out, total = left_truncate_with_bos(ids, 5, _BOS, 20)
        keep = 20 - 5 - 1  # 预留 1 个 BOS 位
        assert total == 105  # 截断前总长（调用方据此告警）
        assert len(out) <= 20
        assert out[0] == _BOS  # BOS 重新前置
        assert out[1:] == ids[-keep:]  # 保尾：末尾 token 原样保留

    def test_without_bos(self):
        ids = list(range(2, 102))
        out, total = left_truncate_with_bos(ids, 5, None, 20)
        assert out == ids[-(20 - 5):]
        assert len(out) == 15 and total == 105


# ---------------------------------------------------------------------------
# 2. SFTDataset：无先行右截断，超长保护保尾
# ---------------------------------------------------------------------------

class TestSFTDataset:
    def test_overlong_prompt_tail_preserved(self):
        """prompt > _MAX_LEN：全序列末 token（目标话语尾部）必须保留（R2-m4 回归锁定）。

        伪 tokenizer 忠实模拟 HF 语义——若上游仍传 truncation=True，尾部会
        被先行砍掉，本测试即失败。
        """
        from debate_erc.agents.prompt_builder import build_classify_prompt

        long_utt = " ".join(f"w{i}" for i in range(2100))  # 远超 2048 token
        sample = DialogueSample(
            context=DialogueContext(
                utterances=("hi", long_utt), speakers=("A", "B"),
                target_idx=1, history_window=1, dialogue_id="long",
            ),
            gold_label="anger",
        )
        tok = FakeTokenizer()
        prompt = build_classify_prompt(sample.context, MELD_SCHEMA)
        full_ids = tok(prompt, add_special_tokens=True, truncation=False).input_ids
        assert len(full_ids) + 1 > 2048  # 前置条件：样本确实超长
        dataset = _SFTDataset([sample], tok, MELD_SCHEMA)
        ex = dataset[0]
        assert len(ex["input_ids"]) <= 2048
        prompt_ids = ex["input_ids"][: ex["prompt_len"]]
        assert prompt_ids[0] == _BOS            # BOS 重新前置
        assert prompt_ids[-1] == full_ids[-1]   # 末 token（保尾）未被先行右截断砍掉
        assert ex["input_ids"][-1] == 10        # answer（"anger"）在最末且未被截掉
        assert ex["label_pos"] == ex["prompt_len"]

    def test_normal_sample_shape(self):
        dataset = _SFTDataset(make_samples(1), FakeTokenizer(), MELD_SCHEMA)
        ex = dataset[0]
        # answer " anger\n" 空白切词 → 1 个 token
        assert ex["prompt_len"] + 1 == len(ex["input_ids"])
        assert ex["label_class"] == MELD_SCHEMA.index("anger")


# ---------------------------------------------------------------------------
# 3. SFTTrainer.train
# ---------------------------------------------------------------------------

class TestSFTTrainerTrain:
    def _trainer(self, tmp_path, **cfg_kwargs) -> SFTTrainer:
        params = dict(
            epochs=1, batch_size=2, grad_accum=1, warmup_ratio=0.0, save_every_epoch=False
        )
        params.update(cfg_kwargs)
        cfg = TrainingConfig(**params)
        return SFTTrainer(TinyLM(), FakeTokenizer(), MELD_SCHEMA, cfg, tmp_path)

    def test_one_epoch_steps_eval_and_mode_restore(self, tmp_path):
        trainer = self._trainer(tmp_path)
        seen_modes = []

        def eval_fn(model):
            seen_modes.append(model.training)  # eval 期间应为 False
            return 0.5

        summary = trainer.train(make_samples(4), eval_fn=eval_fn)
        assert summary["steps"] == 2  # ceil(4/2) 个 batch / accum=1
        assert summary["epochs_done"] == 1
        assert seen_modes == [False]  # eval_fn 调用前 model.eval()（S5/S6 契约）
        assert summary["eval_history"] == [0.5]
        assert summary["final_loss"] == summary["final_loss"]  # 有限值（非 NaN）

    def test_budget_truncation_skips_epoch_wrapup(self, tmp_path):
        """steps_budget/max_steps 截断：不落盘 checkpoint、不计入 epochs_done。"""
        trainer = self._trainer(
            tmp_path, epochs=3, batch_size=1, max_steps=1, save_every_epoch=True
        )
        summary = trainer.train(make_samples(2))
        assert summary["steps"] == 1
        assert summary["epochs_done"] == 0  # 截断 epoch 不计数
        assert not (tmp_path / "checkpoint_epoch1").exists()

    def test_warmup_linear_decay_reaches_zero(self, tmp_path):
        """S5 warmup：4 步预算、warmup_ratio=0.5 → 末步 lr 衰减到 0。"""
        cfg = TrainingConfig(
            epochs=1, batch_size=1, grad_accum=1, warmup_ratio=0.5,
            save_every_epoch=False, lr=1e-3,
        )
        model = TinyLM()
        trainer = SFTTrainer(model, FakeTokenizer(), MELD_SCHEMA, cfg, tmp_path)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
        )
        summary = trainer.train(make_samples(4), optimizer=optimizer)
        assert summary["steps"] == 4
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    def test_tuple_samples_rejected(self, tmp_path):
        trainer = self._trainer(tmp_path)
        bad = [(make_samples(1)[0], 3)]  # (sample, target) 元组 → 明确拒绝
        with pytest.raises(TypeError, match="multitask_scheduler"):
            trainer.train(bad)


# ---------------------------------------------------------------------------
# 4. DPOTrainer
# ---------------------------------------------------------------------------

def _peft_tiny():
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        TinyLM(), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["head"])
    )


class TestDPOTrainer:
    def test_non_peft_model_rejected(self, tmp_path):
        """S3 契约：非 PeftModel 直接 raise（禁止静默退化 ref=policy）。"""
        with pytest.raises(RuntimeError, match="PeftModel"):
            DPOTrainer(
                TinyLM(), FakeTokenizer(), TrainingConfig(), tmp_path
            )

    def test_train_saves_adapter_only(self, tmp_path):
        model = _peft_tiny()
        cfg = TrainingConfig(
            dpo_epochs=1, dpo_batch_size=1, dpo_grad_accum=1, dpo_lr=1e-4, max_steps=0
        )
        trainer = DPOTrainer(model, FakeTokenizer(), cfg, tmp_path)
        summary = trainer.train(make_pairs(1))
        assert summary["steps"] == 1
        epoch_dir = tmp_path / "dpo_epoch1"
        assert (epoch_dir / "adapter_model.safetensors").exists()
        # 只落 adapter：不出现全量权重文件
        assert not (epoch_dir / "model.safetensors").exists()
        assert not (epoch_dir / "pytorch_model.bin").exists()
        assert summary["implicit_acc"] >= 0.0

    def test_max_steps_truncation_skips_epoch_save(self, tmp_path):
        """R2-m6：max_steps 截断的 epoch 不落盘 dpo_epoch{n}（与 SFT 同口径）。"""
        model = _peft_tiny()
        cfg = TrainingConfig(
            dpo_epochs=2, dpo_batch_size=1, dpo_grad_accum=1, dpo_lr=1e-4, max_steps=1
        )
        trainer = DPOTrainer(model, FakeTokenizer(), cfg, tmp_path)
        summary = trainer.train(make_pairs(2))
        assert summary["steps"] == 1
        assert not (tmp_path / "dpo_epoch1").exists()
        assert not (tmp_path / "dpo_epoch2").exists()


# ---------------------------------------------------------------------------
# 5. MultitaskScheduler
# ---------------------------------------------------------------------------

class TestMultitaskScheduler:
    def test_zero_weight_trains_no_aux_steps(self, tmp_path):
        """R2-m5：weight=0 → 辅助任务预算 0，完全不训练。"""
        cfg = TrainingConfig(
            epochs=1, batch_size=1, grad_accum=1, warmup_ratio=0.0, save_every_epoch=False
        )
        model = TinyLM()
        main_trainer = SFTTrainer(model, FakeTokenizer(), MELD_SCHEMA, cfg, tmp_path)
        aux_trainer = SFTTrainer(model, FakeTokenizer(), MELD_SCHEMA, cfg, tmp_path / "aux")
        scheduler = MultitaskScheduler(
            main_trainer, {"aux": (aux_trainer, 0.0)},
            DummyAdapterManager(), switch_steps=10,
        )
        summary = scheduler.train(make_samples(2), {"aux": make_samples(2)})
        assert summary["main_steps"] == 2
        assert summary["aux_steps"]["aux"] == 0
