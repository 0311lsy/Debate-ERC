"""run_id 唯一性与 S1 编码契约测试（审查修复 E 任务 4）。

- 唯一性：全部实验/消融配置 × seeds {42, 43, 44} 的 run_id 两两唯一
  （run 目录以 run_id 命名，冲突会互相覆盖 results.json / predictions.jsonl）；
- S1 关键示例：真实配置文件加载 → make_run_id 的端到端完整映射
  （A 组 test_config_invariants.py 已覆盖 make_run_id 编码函数本身的逻辑
  与唯一性断言，本文件从「配置文件 → run_id」的端到端口径补充关键示例）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.config import load_config, make_run_id  # noqa: E402

EXPERIMENT_DIR = PROJECT_ROOT / "configs" / "experiments"
ALL_CONFIGS = sorted(EXPERIMENT_DIR.glob("*.yaml")) + sorted(
    (EXPERIMENT_DIR / "ablations").glob("*.yaml")
)
SEEDS = (42, 43, 44)

assert ALL_CONFIGS, f"未发现实验配置: {EXPERIMENT_DIR}"


def _load(cfg_name: str):
    """按配置文件名（不含扩展名）加载 ExperimentConfig。"""
    for cfg_path in ALL_CONFIGS:
        if cfg_path.stem == cfg_name:
            return load_config(cfg_path, PROJECT_ROOT)
    raise FileNotFoundError(f"配置不存在: {cfg_name}")


def test_run_id_pairwise_unique_all_configs_all_seeds():
    """全部配置 × 全部 seed 的 run_id 两两唯一（S1）。"""
    seen: dict[str, str] = {}  # run_id -> "配置名@seed"
    for cfg_path in ALL_CONFIGS:
        config = load_config(cfg_path, PROJECT_ROOT)
        for seed in SEEDS:
            run_id = make_run_id(config, seed)
            assert run_id not in seen, (
                f"run_id 冲突: {run_id}"
                f"（{cfg_path.stem}@{seed} 与 {seen[run_id]}）"
            )
            seen[run_id] = f"{cfg_path.stem}@{seed}"


def test_run_id_seed_suffix_encoded():
    """seed 段固定编码为 _s{seed}，同配置不同 seed 不产生相同 run_id。"""
    config = _load("debate_erc_full")
    for seed in SEEDS:
        run_id = make_run_id(config, seed)
        assert run_id.endswith(f"_s{seed}"), f"seed 段缺失: {run_id}"
        assert f"_s{seed + 1}" not in run_id


def test_s1_key_examples_full_run_ids():
    """S1 关键示例：真实配置 → 完整 run_id（A-D 组报告口径逐字符核对）。"""
    cases = {
        # 主方法：debate/adaptive 省略 name 段；auxi 区分强度辅助任务
        # Layer 1：class_balanced=true 现在显式编码 cb 段（变量不得漏编码）
        "debate_erc_full": "debate_full_route_auxi_syn_focal_cb_dpo_s42",
        # 消融：单 Agent + CoT → name + reason，空 parts
        "no_debate": "no_debate_reason_s42",
        # 消融：speaker_task 替换 intensity → auxs 段（与 auxi 区分）
        "aux_speaker": "debate_full_route_auxs_syn_focal_cb_dpo_s42",
        # 消融：双辅助任务 → auxi_auxs 段（按声明顺序）
        "aux_three_task": "debate_full_route_auxi_auxs_syn_focal_cb_dpo_s42",
        # IEMOCAP：非 meld 数据集段进入 run_id（S1 的 _{dataset} 规则）；
        # R2-M1 修复后该配置关闭 synthesis（IEMOCAP 4 类无 fear/disgust）→ 无 syn 段
        "iemocap_debate_full": "iemocap_debate_full_route_auxi_focal_cb_dpo_s42",
    }
    for cfg_name, expected in cases.items():
        config = _load(cfg_name)
        assert make_run_id(config, 42) == expected, (
            f"{cfg_name} 的 run_id 与 S1 契约不符："
            f"实际 {make_run_id(config, 42)!r}，期望 {expected!r}"
        )


def test_s1_aux_tag_segments():
    """辅助任务段细分检查：auxs 含 speaker_task 标签、auxi_auxs 双段按序。"""
    speaker = make_run_id(_load("aux_speaker"), 42)
    assert "auxs" in speaker.split("_"), f"aux_speaker 缺 auxs 段: {speaker}"
    assert "auxi" not in speaker.split("_"), (
        f"aux_speaker 的 intensity 已关闭，不应含 auxi 段: {speaker}"
    )
    three_task = make_run_id(_load("aux_three_task"), 42)
    parts = three_task.split("_")
    assert "auxi" in parts and "auxs" in parts, (
        f"aux_three_task 缺双辅助段: {three_task}"
    )
    assert parts.index("auxi") < parts.index("auxs"), (
        f"多辅助任务段须按 aux_tasks 声明序（auxi_auxs）: {three_task}"
    )


def test_s1_iemocap_dataset_segment():
    """iemocap_debate_full 含 iemocap 数据集段，且 debate 主方法不重复 name 段。"""
    run_id = make_run_id(_load("iemocap_debate_full"), 42)
    assert run_id.startswith("iemocap_debate_"), (
        f"非 meld 数据集须以 dataset 段开头（其后接 method）: {run_id}"
    )
    assert "iemocap" in run_id.split("_"), f"缺 iemocap 段: {run_id}"
    # dataset 段只出现一次（debate 省略 name，name 不再重复 iemocap 前缀语义）
    assert run_id.split("_").count("iemocap") == 1, f"dataset 段重复: {run_id}"
