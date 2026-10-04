"""配置不变量测试（红线 #1/#2 + 装配校验，CI 强制）。

覆盖：
1. 全部实验/消融配置可加载，且 LoRA 与基线完全一致（lora 消融组豁免）；
2. 输出目录禁止 /tmp、/var/tmp、/dev/shm（红线 #2）；
3. seed 集 ⊇ {42, 43, 44}（≥3 seeds）；
4. 无效组合（reask 无 debate / dpo 无偏好对来源 / dpo 无 LoRA /
   reask.mode 未实现 / self_refine 未实现 / reask 阈值倒挂）被装配器拒绝；
5. run_id 编码规范（S1：{name}[_{dataset}]_{method}[_{parts}]_s{seed}）与全配置唯一性；
6. 审查修复后默认值（low_confidence=0.75 / eval_every_steps=0）。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debate_erc.config import (
    AuxTaskConfig,
    DataConfig,
    DebateConfig,
    ExperimentConfig,
    FewShotConfig,
    LoRAConfig,
    ReaskConfig,
    ReasoningConfig,
    RoutingConfig,
    TrainingConfig,
    load_config,
    make_run_id,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "configs" / "experiments"
ALL_CONFIGS = sorted(EXPERIMENT_DIR.glob("*.yaml")) + sorted(
    (EXPERIMENT_DIR / "ablations").glob("*.yaml")
)
# LoRA 消融豁免组（V3 6.4：r / 目标模块 / 无微调 本身就是消融变量）
LORA_ABLATION_NAMES = {"lora_r16", "lora_r32", "lora_qkv", "no_finetune"}

assert ALL_CONFIGS, f"未发现实验配置: {EXPERIMENT_DIR}"


@pytest.fixture(scope="module")
def baseline_lora():
    return load_config(EXPERIMENT_DIR / "base_sft.yaml", PROJECT_ROOT).lora


@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_config_loads(cfg_path):
    load_config(cfg_path, PROJECT_ROOT)  # 加载即通过 validate_config


@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_lora_matches_baseline(cfg_path, baseline_lora):
    if cfg_path.stem in LORA_ABLATION_NAMES:
        pytest.skip(f"LoRA 消融豁免组: {cfg_path.stem}")
    config = load_config(cfg_path, PROJECT_ROOT)
    assert config.lora == baseline_lora, (
        f"{cfg_path.stem} 的 LoRA 配置与基线不一致（红线 #1：防微调增益污染）"
    )


@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_output_dir_not_tmp(cfg_path):
    config = load_config(cfg_path, PROJECT_ROOT)
    assert not config.output_dir.startswith(("/tmp", "/var/tmp", "/dev/shm")), (
        "红线 #2：输出目录禁止 /tmp、/var/tmp、/dev/shm"
    )


@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_seeds_cover_three(cfg_path):
    config = load_config(cfg_path, PROJECT_ROOT)
    assert {42, 43, 44}.issubset(set(config.seeds)), "seed 集 ⊇ {42,43,44}"


def test_reask_without_debate_rejected():
    config = replace(
        ExperimentConfig(),
        reask=ReaskConfig(enabled=True),
        debate=DebateConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="reask"):
        validate_config(config)


def test_dpo_without_pair_source_rejected():
    config = replace(ExperimentConfig(), training=TrainingConfig(dpo_enabled=True))
    with pytest.raises(ValueError, match="dpo"):
        validate_config(config)


def test_dpo_with_reasoning_source_accepted():
    config = replace(
        ExperimentConfig(),
        reasoning=ReasoningConfig(enabled=True),
        training=TrainingConfig(dpo_enabled=True),
    )
    validate_config(config)  # 不抛异常即通过


def test_aux_three_plus_rejected():
    aux = tuple(
        replace(a, enabled=True)
        for a in ExperimentConfig().aux_tasks
    ) + (type(ExperimentConfig().aux_tasks[0])(name="irony", enabled=True),)
    config = replace(ExperimentConfig(), aux_tasks=aux)
    with pytest.raises(ValueError, match="辅助任务"):
        validate_config(config)


def test_reask_mode_unimplemented_rejected():
    config = replace(
        ExperimentConfig(),
        debate=DebateConfig(enabled=True),
        reask=ReaskConfig(enabled=True, mode="vote"),
    )
    with pytest.raises(ValueError, match="未实现"):
        validate_config(config)


def test_debate_self_refine_rejected():
    config = replace(ExperimentConfig(), debate=DebateConfig(self_refine=True))
    with pytest.raises(ValueError, match="self_refine"):
        validate_config(config)


def test_reask_low_confidence_not_above_threshold_rejected():
    config = replace(
        ExperimentConfig(),
        debate=DebateConfig(enabled=True, confidence_threshold=0.8),
        reask=ReaskConfig(enabled=True, low_confidence=0.75),
    )
    with pytest.raises(ValueError, match="low_confidence"):
        validate_config(config)


def test_dpo_without_lora_rejected():
    config = replace(
        ExperimentConfig(),
        reasoning=ReasoningConfig(enabled=True),
        training=TrainingConfig(dpo_enabled=True),
        lora=LoRAConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="lora"):
        validate_config(config)


@pytest.mark.parametrize("bad_dir", ["/tmp/runs", "/var/tmp/runs", "/dev/shm/runs"])
def test_output_dir_tmp_variants_rejected(bad_dir):
    config = replace(ExperimentConfig(), output_dir=bad_dir)
    with pytest.raises(ValueError, match="output_dir"):
        validate_config(config)


def test_review_default_values():
    config = ExperimentConfig()
    assert config.reask.low_confidence == 0.75
    assert config.training.eval_every_steps == 0


def test_run_id_encoding():
    # base：method=base 省略 method 段；空 parts 无 sft 兜底
    config = replace(ExperimentConfig(), name="base_sft")
    assert make_run_id(config, 42) == "base_sft_s42"
    # 非 meld 数据集 → dataset 段（S1 示例）
    config = replace(config, data=DataConfig(dataset="iemocap"))
    assert make_run_id(config, 42) == "base_sft_iemocap_s42"
    # 单 Agent + CoT 基线：name + reason，空 parts（S1 示例）
    config = replace(
        ExperimentConfig(), name="no_debate", reasoning=ReasoningConfig(enabled=True)
    )
    assert make_run_id(config, 42) == "no_debate_reason_s42"
    # debate 主方法省略 name 段；reask 关闭 → noreask
    debate = replace(
        ExperimentConfig(),
        debate=DebateConfig(enabled=True),
        reask=ReaskConfig(enabled=False),
    )
    assert make_run_id(debate, 43) == "debate_noreask_s43"
    # 全组件：aux 标签区分任务名（auxi/auxs）+ dpo 段（S1 示例）
    full = replace(
        ExperimentConfig(),
        debate=DebateConfig(enabled=True),
        reask=ReaskConfig(enabled=True),
        routing=RoutingConfig(enabled=True),
        aux_tasks=(
            AuxTaskConfig(name="intensity", enabled=True),
            AuxTaskConfig(name="speaker_task", enabled=True),
        ),
        fewshot=FewShotConfig(synthesis_enabled=True, focal_loss_enabled=True),
        training=TrainingConfig(dpo_enabled=True),
    )
    assert make_run_id(full, 42) == "debate_full_route_auxi_auxs_syn_focal_dpo_s42"
    # speaker_task 替换 intensity → 仅 auxs
    speaker_only = replace(
        full,
        aux_tasks=(
            AuxTaskConfig(name="intensity", enabled=False),
            AuxTaskConfig(name="speaker_task", enabled=True),
        ),
    )
    assert make_run_id(speaker_only, 42) == "debate_full_route_auxs_syn_focal_dpo_s42"


def test_run_id_unique_across_all_configs():
    seen: dict[str, str] = {}
    for cfg_path in ALL_CONFIGS:
        config = load_config(cfg_path, PROJECT_ROOT)
        for seed in (42, 43, 44):
            run_id = make_run_id(config, seed)
            assert run_id not in seen, (
                f"run_id 冲突: {run_id}（{cfg_path.stem} 与 {seen[run_id]}）"
            )
            seen[run_id] = cfg_path.stem


def test_non_dict_subconfig_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nlora: 64\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lora"):
        load_config(bad, PROJECT_ROOT)


def test_non_dict_aux_task_element_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\naux_tasks:\n  - intensity\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aux_tasks"):
        load_config(bad, PROJECT_ROOT)


def test_unknown_config_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的配置键"):
        load_config(bad, PROJECT_ROOT)
