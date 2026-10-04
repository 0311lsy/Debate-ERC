"""CLI 入口：python -m debate_erc.pipeline.run --config <yaml> [--seed 42] [--smoke 50]

用法示例（设计报告 6.3）：
    python -m debate_erc.pipeline.run --config configs/experiments/debate_erc_full.yaml --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ..config import load_config, validate_config
from ..utils.logging import get_logger
from .runner import ExperimentRunner

logger = get_logger("debate_erc.pipeline.run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debate-ERC 实验入口")
    parser.add_argument("--config", required=True, help="实验配置 YAML 路径")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖配置中的 seed（单 seed 运行）；缺省跑配置中的全部 seeds")
    parser.add_argument("--smoke", type=int, default=None,
                        help="覆盖 smoke_samples（>0 时每 split 仅跑前 N 条）")
    parser.add_argument("--output-dir", default=None, help="覆盖输出根目录")
    parser.add_argument("--skip-eval", action="store_true",
                        help="训练并保存 adapter 后立即结束（评估走独立快路径脚本）")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = {}
    if args.smoke is not None:
        overrides["smoke_samples"] = args.smoke
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.skip_eval:
        overrides["skip_eval"] = True
    if overrides:
        config = dataclasses.replace(config, **overrides)
        # CLI 覆盖后重新校验（堵住 --output-dir /tmp 等绕过 load_config 校验的口子）
        validate_config(config)

    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    logger.info("[cli] 配置 %s，seeds=%s", config.name, seeds)

    exit_code = 0
    for seed in seeds:
        try:
            runner = ExperimentRunner(config, seed)
            results = runner.run()
            if results.get("skip_eval"):
                logger.info("[cli] %s 训练完成（skip_eval）", results["run_id"])
            else:
                logger.info(
                    "[cli] %s W-F1=%.4f",
                    results["run_id"], results["metrics_full"]["wf1"],
                )
        except Exception:
            logger.exception("[cli] seed=%d 运行失败", seed)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
