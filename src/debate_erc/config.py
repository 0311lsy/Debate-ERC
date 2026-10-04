"""配置对象：单一事实来源（设计报告 5.5）。

- 全部超参数收敛到 dataclass，代码不出现魔法数字
- YAML 支持 `extends` 继承：消融配置只写 diff，天然保证 LoRA 等不变量一致
- 输出目录强制 home 持久路径（禁止 /tmp、/var/tmp、/dev/shm，红线 #2）
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# 子配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    enabled: bool = True  # false = 无微调 prompting 基线


@dataclass(frozen=True)
class DebateConfig:
    enabled: bool = False
    max_rounds: int = 3
    confidence_threshold: float = 0.6
    agent2: str = "same"  # same | hetero
    hetero_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    self_refine: bool = False


@dataclass(frozen=True)
class ReasoningConfig:
    enabled: bool = False  # 推理链注入（ReasoningStage / ReasoningHint）


@dataclass(frozen=True)
class ReaskConfig:
    enabled: bool = False  # 依赖 debate；默认关闭，避免无辩论配置误开兜底
    max_reask: int = 2
    mode: str = "conclude"  # 仅 conclude 已实现（all/vote 校验拒绝）
    vote_n: int = 3
    low_confidence: float = 0.75  # 必须大于 debate.confidence_threshold（validate_config 强制）


@dataclass(frozen=True)
class AuxTaskConfig:
    name: str  # intensity | speaker_task（irony 未实现，validate_config 拒绝）
    enabled: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    loss_weight: float = 0.3
    switch_steps: int = 100


@dataclass(frozen=True)
class FewShotConfig:
    synthesis_enabled: bool = False
    synthesis_ratio: float = 1.8
    focal_loss_enabled: bool = False
    focal_gamma: float = 2.0
    class_balanced: bool = False


@dataclass(frozen=True)
class RoutingConfig:
    enabled: bool = False
    oracle_mode: bool = False
    short_threshold: int = 60      # 字符数 ≤ → PERCEIVE 候选
    long_threshold: int = 150      # 字符数 ≥ → DELIBERATE 候选
    minority_boost: bool = True    # 含少数类情感词 → 升级到 DELIBERATE


@dataclass(frozen=True)
class TrainingConfig:
    sft_enabled: bool = True
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    dpo_enabled: bool = False
    dpo_beta: float = 0.1
    dpo_epochs: int = 3
    dpo_batch_size: int = 2
    dpo_grad_accum: int = 16
    dpo_lr: float = 5e-6
    save_every_epoch: bool = True
    eval_every_steps: int = 0  # 0 = 关闭（默认关闭，避免默认开启拖慢训练）
    max_steps: int = 0  # 0 = 不截断（smoke run 可设小值）


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "meld"  # meld | iemocap
    raw_path: str = "data/raw"
    processed_dir: str = "data/processed"
    history_window: int = 5  # 目标话语前 N 轮上下文


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 200
    do_sample: bool = True
    # Layer 0 审稿修复：dev/test 推理（偏好对生成与最终评估）默认贪心解码，
    # 保证指标可复现、组间差异不被温度采样噪声污染（训练不经过该路径）。
    eval_do_sample: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "base"
    backbone: str = "NousResearch/Llama-2-7b-hf"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    reask: ReaskConfig = field(default_factory=ReaskConfig)
    aux_tasks: tuple[AuxTaskConfig, ...] = (
        AuxTaskConfig(name="intensity"),
        AuxTaskConfig(name="speaker_task", enabled=False),
    )
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seeds: tuple[int, ...] = (42, 43, 44)
    output_dir: str = "outputs/runs"
    smoke_samples: int = 0  # >0 时仅跑前 N 条（冒烟测试）
    skip_eval: bool = False  # 训练+adapter 保存后即返回（评估由独立快路径脚本做）


# ---------------------------------------------------------------------------
# 通用 dict → dataclass 构造
# ---------------------------------------------------------------------------

_TYPE_CACHE: dict[str, type] = {c.__name__: c for c in [
    LoRAConfig, DebateConfig, ReasoningConfig, ReaskConfig, AuxTaskConfig,
    FewShotConfig, RoutingConfig, TrainingConfig, DataConfig, GenerationConfig,
]}


def _resolve_field_type(f) -> tuple[str, type | None]:
    """解析字段注解（from __future__ import annotations 使注解全为字符串）。

    返回 (kind, cls)：
    - ("dataclass", cls)：字段本身是 dataclass（如 lora: LoRAConfig）
    - ("dataclass_seq", cls)：dataclass 序列（如 aux_tasks: tuple[AuxTaskConfig, ...]）
    - ("other", None)：标量 / 标量序列
    """
    tp = f.type
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return "dataclass", tp
    if isinstance(tp, str):
        if tp in _TYPE_CACHE and dataclasses.is_dataclass(_TYPE_CACHE[tp]):
            return "dataclass", _TYPE_CACHE[tp]
        m = re.match(r"^(?:tuple|list)\[([A-Za-z_]\w*)", tp)
        if m and m.group(1) in _TYPE_CACHE:
            return "dataclass_seq", _TYPE_CACHE[m.group(1)]
    return "other", None


def _from_dict(cls: type, payload: dict[str, Any]) -> Any:
    """按 dataclass 字段过滤未知键并递归构造（unknown key 直接报错，防配置拼错静默失效）。

    子 dataclass 字段收到非 dict 值、dataclass 序列收到非 dict 元素时立即 ValueError
    （携带字段名与实际类型），避免错误类型静默透传到运行期。
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(payload) - set(fields)
    if unknown:
        raise ValueError(f"{cls.__name__} 不支持的配置键: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, val in payload.items():
        f = fields[key]
        kind, sub_cls = _resolve_field_type(f)
        if kind == "dataclass":
            if not isinstance(val, dict):
                raise ValueError(
                    f"{cls.__name__}.{key} 需要 dict（{sub_cls.__name__}），"
                    f"实际收到 {type(val).__name__}: {val!r}"
                )
            kwargs[key] = _from_dict(sub_cls, val)
        elif kind == "dataclass_seq":
            if not isinstance(val, (list, tuple)):
                raise ValueError(
                    f"{cls.__name__}.{key} 需要 {sub_cls.__name__} 的 list，"
                    f"实际收到 {type(val).__name__}: {val!r}"
                )
            elems = []
            for v in val:
                if not isinstance(v, dict):
                    raise ValueError(
                        f"{cls.__name__}.{key} 的元素需要 dict（{sub_cls.__name__}），"
                        f"实际收到 {type(v).__name__}: {v!r}"
                    )
                elems.append(_from_dict(sub_cls, v))
            kwargs[key] = tuple(elems)  # 注解口径统一为 tuple
        elif isinstance(f.type, str) and f.type.startswith("tuple["):
            # 标量 tuple 注解（如 seeds: tuple[int, ...]）：拒绝字符串等非序列
            # 标量（R2-m1：YAML seeds: "42" 曾被静默透传，run.py 拆成单字符 seed），
            # 并按注解元素类型强转，转换失败在配置期显式报错
            if not isinstance(val, (list, tuple)):
                raise ValueError(
                    f"{cls.__name__}.{key} 需要 list（标量序列），"
                    f"实际收到 {type(val).__name__}: {val!r}"
                )
            elem_cls = _scalar_tuple_elem_type(f.type)
            if elem_cls is not None:
                coerced = []
                for v in val:
                    try:
                        coerced.append(elem_cls(v))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{cls.__name__}.{key} 元素需为 {elem_cls.__name__}: {v!r}"
                        ) from exc
                kwargs[key] = tuple(coerced)
            else:
                kwargs[key] = tuple(val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


_SCALAR_TUPLE_RE = re.compile(r"^tuple\[(int|float|str)(?:, ?\.\.\.)?\]$")


def _scalar_tuple_elem_type(annotation: str) -> type | None:
    """解析标量 tuple 注解的元素类型（tuple[int, ...] → int；未知 → None 不强转）。"""
    m = _SCALAR_TUPLE_RE.match(annotation.strip())
    return {"int": int, "float": float, "str": str}[m.group(1)] if m else None


# ---------------------------------------------------------------------------
# YAML 加载（支持 extends 继承）
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(base[key], val)
        else:
            out[key] = val
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是映射: {path}")
    return payload


def load_config(path: str | Path, project_root: str | Path | None = None) -> ExperimentConfig:
    """加载实验配置；`extends` 相对当前文件解析，可链式继承。"""
    path = Path(path).resolve()
    if project_root is None:
        project_root = path.parent
        while project_root != project_root.parent and not (project_root / "configs").is_dir():
            project_root = project_root.parent
    payload = {}
    chain: list[Path] = []
    cur = path
    while True:
        chain.append(cur)
        raw = _load_yaml(cur)
        parent = raw.pop("extends", None)
        payload = _deep_merge(raw, payload) if payload else raw
        if parent is None:
            break
        cur = (cur.parent / parent).resolve()
        if not cur.exists():  # 尝试相对 configs 根
            alt = (Path(project_root) / "configs" / parent).resolve()
            if alt.exists():
                cur = alt
            else:
                raise FileNotFoundError(f"extends 目标不存在: {parent}")
        if cur in chain:
            raise ValueError(f"配置继承出现环: {chain}")
    # aux_tasks 是列表，特殊处理：extends 不做深合并，整体覆盖；
    # chain 顺序为 叶→根，reversed 后根先处理、叶最后处理 → 最派生配置生效
    for p in reversed(chain):
        raw = _load_yaml(p)
        if "aux_tasks" in raw:
            payload["aux_tasks"] = raw["aux_tasks"]
    if "name" not in payload:
        payload["name"] = path.stem
    config = _from_dict(ExperimentConfig, payload)
    validate_config(config)
    return config


# ---------------------------------------------------------------------------
# 装配约束校验（设计报告 7.2）
# ---------------------------------------------------------------------------


def validate_config(config: ExperimentConfig) -> None:
    errors: list[str] = []
    # 红线 #2：输出目录禁止临时文件系统（易失 + 权限宽松，容器重建即丢）
    _TMP_DIR_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm")
    if config.output_dir.startswith(_TMP_DIR_PREFIXES):
        errors.append(
            f"output_dir 禁止使用临时目录 {_TMP_DIR_PREFIXES}: {config.output_dir}"
        )
    # reask 依赖 debate
    if config.reask.enabled and not config.debate.enabled:
        errors.append("reask.enabled=true 需要 debate.enabled=true（兜底依赖辩论分歧点）")
    # reask.mode 仅 conclude 已实现
    if config.reask.mode != "conclude":
        errors.append(
            f"reask.mode={config.reask.mode!r} 未实现（当前仅支持 conclude）"
        )
    # debate.self_refine 未实现
    if config.debate.self_refine:
        errors.append("debate.self_refine=true 未实现（自精炼轮次暂未接入辩论循环）")
    # reask 兜底触发阈值必须高于辩论收敛阈值，否则二次提交永不触发
    if config.reask.enabled and config.reask.low_confidence <= config.debate.confidence_threshold:
        errors.append(
            f"reask.enabled=true 要求 reask.low_confidence ({config.reask.low_confidence}) "
            f"大于 debate.confidence_threshold ({config.debate.confidence_threshold})"
        )
    # dpo 需要偏好对来源（设计报告 7.2：debate 或 reasoning 均可作为来源）
    if config.training.dpo_enabled and not (
        config.debate.enabled or config.reasoning.enabled
    ):
        errors.append(
            "dpo_enabled=true 需要 debate.enabled=true 或 reasoning.enabled=true（偏好对来源）"
        )
    # dpo 必须走 LoRA（7B 全参 DPO 必然 OOM）
    if config.training.dpo_enabled and not config.lora.enabled:
        errors.append(
            "dpo_enabled=true 需要 lora.enabled=true（7B 全参 DPO 必然 OOM）"
        )
    # 3+ 辅助任务必须任务特定 adapter（本实现即任务特定，但 r 容量预警）
    enabled_aux = [a for a in config.aux_tasks if a.enabled]
    if len(enabled_aux) >= 3:
        errors.append("3+ 辅助任务负迁移风险高（V3 5.5），请拆分为独立消融配置")
    # 辅助任务名白名单（R2-m2）：未实现的任务在配置期拒绝，
    # 而非等模型加载后（甚至 lora.enabled=false 时静默无操作）才暴露
    _IMPLEMENTED_AUX_TASKS = ("intensity", "speaker_task")
    unknown_aux = [a.name for a in enabled_aux if a.name not in _IMPLEMENTED_AUX_TASKS]
    if unknown_aux:
        errors.append(
            f"辅助任务未实现: {unknown_aux}（当前仅支持 {list(_IMPLEMENTED_AUX_TASKS)}）"
        )
    if config.debate.agent2 not in ("same", "hetero"):
        errors.append(f"debate.agent2 非法: {config.debate.agent2}")
    if errors:
        raise ValueError("配置装配校验失败:\n  - " + "\n  - ".join(errors))


# 辅助任务 → run_id 标签（须区分任务名；多任务按 aux_tasks 声明顺序以 _ 连接）
_AUX_TAG_BY_NAME: dict[str, str] = {
    "intensity": "auxi",
    "speaker_task": "auxs",
}


def make_run_id(config: ExperimentConfig, seed: int) -> str:
    """run_id 编码规范（审查修复 S1）：

    `{name}[_{dataset} 若非 meld]_{method}[_{parts 以 _ 连接}]_s{seed}`

    - method 判定顺序：debate→`debate`，routing→`adaptive`，reasoning/DPO→`reason`，
      fewshot→`few`，否则 `base`；`base` 时省略 method 段，`debate`/`adaptive`
      主方法省略 name 段（与旧编码 `{method}_{components}_s{seed}` 保持兼容）
    - parts：debate 开→`full`/`noreask`（+`hetero`）、routing 开→`route`（+`oracle`）、
      辅助任务按名区分（intensity→`auxi`，speaker_task→`auxs`，多任务以 _ 连接）、
      `syn`/`focal`、`dpo`、`nft`/`r{r}`/`qkvo`；parts 为空时无该段
    - 示例：`no_debate_reason_s42`、`debate_full_route_auxi_auxs_syn_focal_dpo_s42`、
      `base_sft_iemocap_s42`
    """
    method = "base"
    if config.debate.enabled:
        method = "debate"
    elif config.routing.enabled:
        method = "adaptive"
    elif config.reasoning.enabled or config.training.dpo_enabled:
        method = "reason"
    elif config.fewshot.synthesis_enabled or config.fewshot.focal_loss_enabled:
        method = "few"
    parts: list[str] = []
    if config.debate.enabled:
        parts.append("full" if config.reask.enabled else "noreask")
        if config.debate.agent2 == "hetero":
            parts.append("hetero")
    if config.routing.enabled:
        parts.append("route")
        if config.routing.oracle_mode:
            parts.append("oracle")
    if any(a.enabled for a in config.aux_tasks):
        parts.extend(
            _AUX_TAG_BY_NAME.get(a.name, f"aux_{a.name}")
            for a in config.aux_tasks
            if a.enabled
        )
    if config.fewshot.synthesis_enabled:
        # 编码合成配比：默认 1.8 省略（兼容旧 run_id），非默认显式编码（syn12=1.2x）
        tag = "syn"
        ratio = config.fewshot.synthesis_ratio
        if abs(ratio - 1.8) > 1e-6:
            tag += f"{ratio:.1f}".replace(".", "")
        parts.append(tag)
    if config.fewshot.focal_loss_enabled:
        parts.append("focal")
    if config.fewshot.class_balanced:
        parts.append("cb")
    if config.training.dpo_enabled:
        parts.append("dpo")
    if not config.lora.enabled:
        parts.append("nft")
    elif config.lora.r != 64:
        parts.append(f"r{config.lora.r}")
    if len(config.lora.target_modules) > 2:
        parts.append("qkvo")
    segments: list[str] = []
    if method not in ("debate", "adaptive"):
        segments.append(config.name)
    if config.data.dataset != "meld":
        segments.append(config.data.dataset)
    if method != "base":
        segments.append(method)
    if parts:
        segments.append("_".join(parts))
    segments.append(f"s{seed}")
    return "_".join(segments)
