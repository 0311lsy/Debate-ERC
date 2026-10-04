"""第二轮审查修复的回归测试（R2-M1/M3/m1/m2/m3/m17）。

覆盖：
- M1: apply_fewshot_synthesis 目标类缺失时显式 raise（IEMOCAP 静默失效防线）；
- M3: DialogueContext.render 目标行锁定（居中目标不丢失；末位目标等价保尾）；
- m1: 标量 tuple 字段类型校验（seeds: "42" 配置期拒绝 / 元素强转）；
- m2: validate_config 辅助任务名白名单（irony 拒绝）；
- m3: long_threshold 边界语义统一（DifficultyRouter 与 TemplateRouter 同用 ≥）；
- m17: 脚本核心行为（配置发现、smoke 过滤聚合、预测对齐、mean_std）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.config import (  # noqa: E402
    AuxTaskConfig,
    ExperimentConfig,
    _from_dict,
    validate_config,
)
from debate_erc.data import apply_fewshot_synthesis  # noqa: E402
from debate_erc.data.schemas import DialogueContext  # noqa: E402
from debate_erc.reasoning.template_router import TemplateRouter  # noqa: E402
from debate_erc.routing.difficulty_router import DifficultyRouter, RouteMode  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# R2-M1：synthesis 目标类缺失 → 配置期显式失败
# ---------------------------------------------------------------------------

class TestSynthesisGuard:
    def test_raises_when_targets_absent(self):
        from debate_erc.data.schemas import DialogueSample

        samples = [
            DialogueSample(
                context=DialogueContext(
                    utterances=("hi",), speakers=("A",),
                    target_idx=0, history_window=1, dialogue_id=f"d{i}",
                ),
                gold_label="neutral",
            )
            for i in range(3)
        ]
        with pytest.raises(ValueError, match="synthesis"):
            apply_fewshot_synthesis(samples)

    def test_meld_targets_pass(self):
        """含 fear/disgust 原始样本时正常合成（不 raise）。"""
        from debate_erc.data.schemas import DialogueSample

        samples = []
        for i in range(6):
            label = "fear" if i % 2 == 0 else "disgust"
            samples.append(DialogueSample(
                context=DialogueContext(
                    utterances=("hi", "scared of that"), speakers=("A", "B"),
                    target_idx=1, history_window=1, dialogue_id=f"d{i}",
                ),
                gold_label=label,
            ))
        augmented = apply_fewshot_synthesis(samples, ratio=1.8, seed=42)
        assert len(augmented) > len(samples)
        assert any(s.synthetic for s in augmented)


# ---------------------------------------------------------------------------
# R2-M3：render 目标行锁定
# ---------------------------------------------------------------------------

class TestRenderTargetLineLocked:
    def test_middle_target_not_lost(self):
        """目标行居中且后文超预算：目标行必须保留（实证 R2-M3 场景）。"""
        ctx = DialogueContext(
            utterances=("hist", "TARGET HERE.", "C" * 200, "D"),
            speakers=("A", "B", "C", "D"),
            target_idx=1,
            history_window=1,
        )
        out = ctx.render(max_chars=30)
        assert "TARGET HERE." in out  # 修复前此断言失败（目标行被丢）
        assert "A: hist" not in out   # 预算不足的远侧行被裁剪

    def test_last_target_equivalent_to_tail_keep(self):
        """目标行在末位（生产构造点）：超预算目标行完整保留、前文按预算就近裁剪。"""
        long_utt = "X" * 300
        ctx = DialogueContext(
            utterances=("short", long_utt),
            speakers=("A", "B"),
            target_idx=1,
            history_window=1,
        )
        out = ctx.render(max_chars=50)
        assert out == f"B: {long_utt} [TARGET]"  # 目标行超预算也完整保留

    def test_under_limit_unchanged(self):
        ctx = DialogueContext(
            utterances=("hi", "there"), speakers=("A", "B"),
            target_idx=1, history_window=1,
        )
        assert ctx.render(max_chars=800) == "A: hi\nB: there [TARGET]"


# ---------------------------------------------------------------------------
# R2-m1 / R2-m2：配置期校验
# ---------------------------------------------------------------------------

class TestConfigValidationRound2:
    def test_seeds_scalar_string_rejected(self):
        """YAML seeds: "42"（字符串标量）→ 配置期 ValueError，不再静默透传。"""
        with pytest.raises(ValueError, match="seeds"):
            _from_dict(ExperimentConfig, {"seeds": "42"})

    def test_seeds_string_elements_coerced_to_int(self):
        cfg = _from_dict(ExperimentConfig, {"seeds": ["42", 43]})
        assert cfg.seeds == (42, 43)
        assert all(isinstance(s, int) for s in cfg.seeds)

    def test_seeds_bad_element_rejected(self):
        with pytest.raises(ValueError, match="seeds"):
            _from_dict(ExperimentConfig, {"seeds": ["abc"]})

    def test_unimplemented_aux_task_rejected(self):
        """R2-m2：irony 未实现 → validate_config 拒绝（配置期而非模型加载后）。"""
        cfg = _from_dict(
            ExperimentConfig,
            {"aux_tasks": [{"name": "irony", "enabled": True}]},
        )
        with pytest.raises(ValueError, match="irony"):
            validate_config(cfg)

    def test_implemented_aux_tasks_accepted(self):
        cfg = _from_dict(
            ExperimentConfig,
            {"aux_tasks": [
                {"name": "intensity", "enabled": True},
                {"name": "speaker_task", "enabled": True},
            ]},
        )
        validate_config(cfg)  # 不 raise


# ---------------------------------------------------------------------------
# R2-m3：long_threshold 边界一致（两个 Router 同用 ≥）
# ---------------------------------------------------------------------------

class TestLongThresholdBoundary:
    def test_exact_threshold_routes_consistently(self):
        target = "a" * 150  # 恰等于阈值
        ctx = DialogueContext(
            utterances=("prev line", target),
            speakers=("A", "B"),
            target_idx=1,
            history_window=1,
        )
        mode = DifficultyRouter(long_threshold=150).route(ctx)
        template = TemplateRouter(MELD_SCHEMA, long_utterance_chars=150)
        assert mode == RouteMode.DELIBERATE  # difficulty: length >= 150
        assert template.route(ctx).name == "LONG_UTTERANCE"  # template: >= 150（统一后）


# ---------------------------------------------------------------------------
# R2-m17：脚本核心行为
# ---------------------------------------------------------------------------

class TestScriptLogic:
    def test_discover_configs_filter_and_missing(self):
        mod = _load_script("run_ablation_matrix")
        all_cfgs = mod.discover_configs(None)
        assert all_cfgs and all(p.suffix == ".yaml" for p in all_cfgs)
        picked = mod.discover_configs([all_cfgs[0].stem])
        assert [p.stem for p in picked] == [all_cfgs[0].stem]
        with pytest.raises(SystemExit):
            mod.discover_configs(["no_such_config_xyz"])

    def test_aggregate_collect_runs_filters_smoke_and_broken(self, tmp_path):
        mod = _load_script("aggregate_results")
        good = tmp_path / "base_sft_s42"
        good.mkdir()
        (good / "results.json").write_text(json.dumps({
            "config_name": "base_sft", "seed": 42, "smoke_samples": 0,
            "metrics_full": {"wf1": 0.5, "macro_f1": 0.4},
            "metrics_real_only": {"wf1": 0.5, "macro_f1": 0.4},
        }), encoding="utf-8")
        smoke = tmp_path / "base_sft_s43"
        smoke.mkdir()
        (smoke / "results.json").write_text(json.dumps({
            "config_name": "base_sft", "seed": 43, "smoke_samples": 50,
            "metrics_full": {"wf1": 0.9, "macro_f1": 0.8},
            "metrics_real_only": {"wf1": 0.9, "macro_f1": 0.8},
        }), encoding="utf-8")
        broken = tmp_path / "broken_s42"
        broken.mkdir()
        (broken / "results.json").write_text("{oops", encoding="utf-8")
        by_config, skipped = mod.collect_runs(tmp_path)
        assert skipped == 1  # smoke run 过滤
        assert [r["seed"] for r, _ in by_config["base_sft"]] == [42]  # 损坏文件跳过

    def test_aggregate_mean_std(self):
        mod = _load_script("aggregate_results")
        mean, std = mod.mean_std([0.6, 0.8])
        assert mean == pytest.approx(0.7)
        assert std == pytest.approx(0.02 ** 0.5)  # 样本 std（n-1）= √0.02
        mean, std = mod.mean_std([0.5, 0.5, 0.5])
        assert (mean, std) == (0.5, 0.0)
        assert mod.mean_std([])[0] != mod.mean_std([])[0]  # 空 → NaN

    def test_load_aligned_predictions(self, tmp_path):
        mod = _load_script("aggregate_results")
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir(), dir_b.mkdir()
        rows = [("fear", "fear"), ("anger", "neutral")]
        for d, preds in ((dir_a, ("fear", "anger")), (dir_b, ("fear", "neutral"))):
            with open(d / "predictions.jsonl", "w", encoding="utf-8") as fh:
                for (gold, _p), pred in zip(rows, preds):
                    fh.write(json.dumps({"gold": gold, "pred": pred}) + "\n")
        aligned = mod.load_aligned_predictions(dir_a, dir_b)
        assert aligned is not None
        preds_a, preds_b, golds = aligned
        assert (preds_a, preds_b, golds) == (["fear", "anger"], ["fear", "neutral"],
                                             ["fear", "anger"])
        # gold 错位 → None（防数据版本不一致静默 bootstrap）
        with open(dir_b / "predictions.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"gold": "joyful", "pred": "fear"}) + "\n")
            fh.write(json.dumps({"gold": "anger", "pred": "neutral"}) + "\n")
        assert mod.load_aligned_predictions(dir_a, dir_b) is None
