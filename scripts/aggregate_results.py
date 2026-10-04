#!/usr/bin/env python
"""多 seed 结果汇总（M3 任务 3.2）：均值±std 主表 + 显著性检验 + 双口径输出。

用法：
    python scripts/aggregate_results.py --run-dir outputs/runs --out results_table.md
                                       [--baseline base_sft] [--main debate_erc_full]

产出（红线 #6：全样本 / 仅真实子集 双列）：
- Markdown 主表：每配置 W-F1 / Macro-F1 均值±std（两种口径）
- 主方法 vs 基线：逐 seed 配对 t 检验 + Cohen's d + bootstrap 95% CI
  （bootstrap 读两 run 的 predictions.jsonl 逐样本对齐，调 evaluation.bootstrap_ci）
- 辩论过程指标汇总（收敛率 / 错误共识率 / 兜底触发率 / 兜底纠错率，对齐 V3 6.5 目标）

冒烟 run（results.json 的 smoke_samples > 0）在 collect 阶段整体过滤，
计入 skipped_smoke 并在报告头部注明，不参与主表与显著性。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import bootstrap_ci, cohens_d, paired_ttest  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.aggregate")


def collect_runs(
    run_dir: Path,
) -> tuple[dict[str, list[tuple[dict, Path]]], int]:
    """按 config_name 聚合 results.json（返回 [(results, run 目录), ...]）。

    smoke_samples > 0 的冒烟 run 被过滤，只累计进 skipped_smoke 统计。
    """
    by_config: dict[str, list[tuple[dict, Path]]] = defaultdict(list)
    skipped_smoke = 0
    for results_file in sorted(run_dir.glob("*/results.json")):
        try:
            with open(results_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("跳过损坏的 results.json: %s (%s)", results_file, exc)
            continue
        # 冒烟 run 不参与汇总（S6：smoke_samples 随结果落盘）
        if int(data.get("smoke_samples") or 0) > 0:
            skipped_smoke += 1
            logger.info("[smoke] 过滤冒烟 run: %s", results_file.parent.name)
            continue
        by_config[data.get("config_name", results_file.parent.name)].append(
            (data, results_file.parent)
        )
    return by_config, skipped_smoke


def load_aligned_predictions(
    run_dir_a: Path, run_dir_b: Path
) -> tuple[list[str], list[str], list[str]] | None:
    """读两 run 的 predictions.jsonl 并逐样本对齐。

    两 run 跑同一 test split → 落盘行序一致；按行对齐并校验逐行 gold
    相等（不等说明样本错位/数据版本不一致），校验失败或文件缺失返回 None。
    返回 (preds_a, preds_b, golds)。
    """
    def load(path: Path) -> list[dict]:
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    try:
        rows_a = load(run_dir_a / "predictions.jsonl")
        rows_b = load(run_dir_b / "predictions.jsonl")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("predictions.jsonl 不可读，跳过 bootstrap: %s", exc)
        return None
    if not rows_a or len(rows_a) != len(rows_b):
        logger.warning(
            "predictions.jsonl 样本数不一致（a=%d, b=%d），跳过 bootstrap",
            len(rows_a), len(rows_b),
        )
        return None
    golds_a = [r["gold"] for r in rows_a]
    golds_b = [r["gold"] for r in rows_b]
    if golds_a != golds_b:
        logger.warning("两 run 的 predictions.jsonl 逐样本 gold 不一致，跳过 bootstrap")
        return None
    return [r["pred"] for r in rows_a], [r["pred"] for r in rows_b], golds_a


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}±{std * 100:.2f}" if not math.isnan(mean) else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="多 seed 汇总 → 主表 + 显著性")
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "runs"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "results_table.md"))
    parser.add_argument("--baseline", default="base_sft", help="显著性对照基线配置名")
    parser.add_argument("--main", default="debate_erc_full", help="主方法配置名")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    by_config, skipped_smoke = collect_runs(run_dir)
    if not by_config:
        logger.error("未发现任何 results.json（%s）", run_dir)
        return 1

    lines: list[str] = [
        "# Debate-ERC 结果汇总",
        "",
        f"> 冒烟 run 过滤：skipped_smoke={skipped_smoke}（smoke_samples>0 的 run 不参与汇总）",
        "",
    ]
    # ---- 主表（双口径，红线 #6）----
    lines += [
        "## 主表（mean±std，n seeds）", "",
        "| 配置 | n | W-F1 (全样本) | W-F1 (仅真实) | Macro-F1 (全样本) | Macro-F1 (仅真实) |",
        "|---|---|---|---|---|---|",
    ]
    wf1_by_config: dict[str, list[float]] = {}
    for name in sorted(by_config):
        runs = [r for r, _ in by_config[name]]
        full_wf1 = [r["metrics_full"]["wf1"] for r in runs]
        real_wf1 = [r["metrics_real_only"]["wf1"] for r in runs]
        full_mf1 = [r["metrics_full"]["macro_f1"] for r in runs]
        real_mf1 = [r["metrics_real_only"]["macro_f1"] for r in runs]
        wf1_by_config[name] = full_wf1
        lines.append(
            f"| {name} | {len(runs)} | {fmt(*mean_std(full_wf1))} | "
            f"{fmt(*mean_std(real_wf1))} | {fmt(*mean_std(full_mf1))} | "
            f"{fmt(*mean_std(real_mf1))} |"
        )

    # ---- 显著性：主方法 vs 基线（逐 seed 配对）----
    lines += ["", "## 显著性检验（主方法 vs 基线，逐 seed 配对）", ""]
    main_runs = by_config.get(args.main, [])
    base_runs = by_config.get(args.baseline, [])
    if main_runs and base_runs:
        seeds_main = {r["seed"]: r["metrics_full"]["wf1"] for r, _ in main_runs}
        seeds_base = {r["seed"]: r["metrics_full"]["wf1"] for r, _ in base_runs}
        common = sorted(set(seeds_main) & set(seeds_base))
        if len(common) >= 2:
            a = [seeds_main[s] for s in common]
            b = [seeds_base[s] for s in common]
            try:
                t, p = paired_ttest(a, b)
                d = cohens_d(a, b)
                lines += [
                    f"- 配对 seeds: {common}",
                    f"- ΔW-F1 = {fmt(*mean_std([x - y for x, y in zip(a, b)]))}",
                    f"- paired t = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}",
                    f"- 显著（p<0.05）: {'是' if p < 0.05 else '否'}",
                ]
            except (RuntimeError, ValueError) as exc:
                lines.append(f"- 检验不可用: {exc}")
        else:
            lines.append(f"- 共同 seed 不足（main={sorted(seeds_main)}, base={sorted(seeds_base)}）")
        # bootstrap 95% CI（逐样本配对，读两 run 的 predictions.jsonl）
        main_dirs = {r["seed"]: d for r, d in main_runs}
        base_dirs = {r["seed"]: d for r, d in base_runs}
        for seed in common:
            aligned = load_aligned_predictions(main_dirs[seed], base_dirs[seed])
            if aligned is None:
                lines.append(
                    f"- seed {seed} bootstrap：不可用（predictions.jsonl 缺失或样本错位）"
                )
                continue
            preds_a, preds_b, golds = aligned
            mean, lo, hi = bootstrap_ci(preds_a, preds_b, golds)
            excludes_zero = lo > 0 or hi < 0
            lines.append(
                f"- seed {seed} bootstrap：ΔW-F1 = {mean:+.4f}，"
                f"95% CI = [{lo:+.4f}, {hi:+.4f}]，"
                f"{'显著（CI 不含 0）' if excludes_zero else '不显著（CI 含 0）'}"
            )
    else:
        lines.append(f"- 缺少 {args.main} 或 {args.baseline} 的结果")

    # ---- 辩论过程指标（V3 6.5 目标核对）----
    lines += ["", "## 辩论过程指标（目标：收敛率≥80% / 错误共识率≤15% / 兜底触发率≤20%）", ""]
    debate_rows = [
        (name, r["debate"]) for name in sorted(by_config)
        for r, _ in by_config[name] if "debate" in r
    ]
    if debate_rows:
        agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for name, d in debate_rows:
            for key in ("convergence_rate", "error_consensus_rate",
                        "reask_trigger_rate", "reask_correction_rate"):
                if key in d:
                    agg[name][key].append(d[key])
        lines += ["| 配置 | 收敛率 | 错误共识率 | 兜底触发率 | 兜底纠错率 |", "|---|---|---|---|---|"]
        for name in sorted(agg):
            row = [name]
            for key in ("convergence_rate", "error_consensus_rate",
                        "reask_trigger_rate", "reask_correction_rate"):
                row.append(f"{agg[name][key] and mean_std(agg[name][key])[0] * 100:.2f}%"
                           if agg[name][key] else "—")
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("（无辩论 run）")

    # ---- 路由 / Oracle ----
    route_rows = [
        (name, r["routing"], r.get("oracle"))
        for name in sorted(by_config) for r, _ in by_config[name]
        if "routing" in r
    ]
    if route_rows:
        lines += ["", "## 路由统计", ""]
        for name, rt, oracle in route_rows:
            dist = rt.get("mode_distribution", {})
            lines.append(
                f"- {name}: 模式分布 "
                f"{ {k: f'{v * 100:.1f}%' for k, v in sorted(dist.items())} }"
            )
            if oracle:
                lines.append(
                    f"  Oracle 上界 W-F1={oracle['upper_bound_wf1'] * 100:.2f}，"
                    f"gap={oracle['gap'] * 100:.2f}"
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("汇总已写入 %s（%d 配置）", out, len(by_config))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
