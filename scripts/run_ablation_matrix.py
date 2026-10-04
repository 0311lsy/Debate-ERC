#!/usr/bin/env python
"""消融矩阵调度（M3 任务 3.1）：逐配置 × 逐 seed，崩溃续跑。

用法：
    python scripts/run_ablation_matrix.py                       # 全部实验+消融 × 配置 seeds
    python scripts/run_ablation_matrix.py --only debate_erc_full,base_sft --seeds 42,43,44
    python scripts/run_ablation_matrix.py --smoke 50            # 每配置 50 条冒烟

续跑机制：目标 run 目录已存在 results.json 则跳过（崩溃 run 重跑，成功 run 不重复）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.config import load_config, make_run_id  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.matrix")

# 冒烟专用输出根（相对项目根解析，经子进程 --output-dir 覆盖），
# 避免冒烟 results.json 落进正式 run 目录、污染 aggregate_results.py 的汇总
SMOKE_OUTPUT_DIR = "outputs/smoke_runs"


def discover_configs(only: list[str] | None) -> list[Path]:
    configs = sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))
    configs += sorted((PROJECT_ROOT / "configs" / "experiments" / "ablations").glob("*.yaml"))
    if only:
        wanted = set(only)
        configs = [c for c in configs if c.stem in wanted]
        missing = wanted - {c.stem for c in configs}
        if missing:
            raise SystemExit(f"未找到配置: {sorted(missing)}")
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="消融矩阵一键调度")
    parser.add_argument("--only", default=None,
                        help="逗号分隔的配置名（缺省全部实验+消融）")
    parser.add_argument("--seeds", default=None, help="逗号分隔，覆盖配置内 seeds")
    parser.add_argument("--smoke", type=int, default=None,
                        help="冒烟样本数（>0 时每 split 仅跑前 N 条；结果写入 "
                             "outputs/smoke_runs（相对项目根），不污染正式 run 目录）")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    only = args.only.split(",") if args.only else None
    configs = discover_configs(only)
    seeds_override = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else None
    )
    logger.info("矩阵：%d 个配置 × seeds=%s", len(configs), seeds_override or "config")

    failures: list[str] = []
    skipped, done = 0, 0
    t0 = time.time()
    for cfg_path in configs:
        config = load_config(cfg_path, project_root=PROJECT_ROOT)
        seeds = seeds_override or list(config.seeds)
        for seed in seeds:
            run_id = make_run_id(config, seed)
            # --smoke 时子进程经 --output-dir 写入冒烟专用目录，
            # 此处续跑判断须与之同口径（查 smoke 目录而非正式 run 目录）
            out = (
                Path(SMOKE_OUTPUT_DIR) if args.smoke is not None
                else Path(config.output_dir)
            )
            run_dir = out if out.is_absolute() else PROJECT_ROOT / out
            run_dir = run_dir / run_id
            if (run_dir / "results.json").exists():
                skipped += 1
                logger.info("[skip] %s（results.json 已存在）", run_id)
                continue
            cmd = [args.python, "-m", "debate_erc.pipeline.run",
                   "--config", str(cfg_path), "--seed", str(seed)]
            if args.smoke is not None:
                cmd += ["--smoke", str(args.smoke),
                        "--output-dir", SMOKE_OUTPUT_DIR]
            logger.info("[run ] %s", " ".join(cmd))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            ret = subprocess.call(cmd, cwd=PROJECT_ROOT, env=env)
            if ret == 0:
                done += 1
            else:
                failures.append(run_id)
                logger.error("[fail] %s 退出码 %d（续跑将自动重试该 run）", run_id, ret)
    logger.info(
        "矩阵完成：成功 %d，跳过 %d，失败 %d（%.1f 分钟）",
        done, skipped, len(failures), (time.time() - t0) / 60,
    )
    if failures:
        logger.error("失败 run：\n  %s", "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
