#!/usr/bin/env python
"""Layer 1 选择性辩论：dev 标定 + test 验证（严格防 test 调参泄漏）。

流程：
1. dev（1109 条）：每条都计算 proponent（标准 classify 贪心+P_A）与
   critic（自检 hint classify 贪心+P_B）四元组 (y_a,p_a,y_b,p_b)，
   本地网格扫描 τ×margin（不重新推理），按 dev W-F1 选最优，并列取触发率低；
2. test（2610 条）：只接受 dev_calibration.json 传入的参数，禁止网格搜索；
   报告 final vs proponent-only 的指标差、触发率、翻转改对/改错数。

所有解码均为贪心（max_new_tokens=8，SFT " label\\n" 口径），
全部结果可复现。单 checkpoint（m0_sft）零训练验证机制有效性。

用法：
    python scripts/eval_selective.py --mode dev
    python scripts/eval_selective.py --mode test   # 自动读取 dev 标定结果
    python scripts/eval_selective.py --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from debate_erc.agents.prompt_builder import build_classify_prompt  # noqa: E402
from debate_erc.config import load_config  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.deliberation import CRITIC_REASONING_HINT  # noqa: E402
from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.models import LLMGenerator, load_backbone  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402
from debate_erc.utils.post_process import parse_label  # noqa: E402

logger = get_logger("debate_erc.scripts.eval_selective")

OUT_DIR = PROJECT_ROOT / "outputs" / "selective"
M0_SFT = PROJECT_ROOT / "outputs/runs/base_sft_s42/adapter_main/main"
CALIB_FILE = OUT_DIR / "dev_calibration.json"

TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]

GEN_KW = {"do_sample": False, "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0}


def collect_records(generator, schema, samples, critic_for_all: bool, gate_tau: float | None):
    """逐条推理并落盘四元组。

    critic_for_all=True（dev）：全部样本计算 critic 视图供网格扫描；
    critic_for_all=False（test）：仅 p_a < gate_tau 的样本计算 critic（省时），
    其余记录 y_b/p_b=None。
    """
    labels = list(schema.labels)
    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt_a = build_classify_prompt(s.context, schema)
        y_a = parse_label(generator.generate(prompt_a, **GEN_KW), schema)
        p_a_dist = generator.score_labels(prompt_a, labels)
        p_a = p_a_dist[y_a]

        need_critic = critic_for_all or (p_a < gate_tau)
        if need_critic:
            prompt_b = build_classify_prompt(
                s.context, schema, reasoning_hint=CRITIC_REASONING_HINT
            )
            y_b = parse_label(generator.generate(prompt_b, **GEN_KW), schema)
            p_b_dist = generator.score_labels(prompt_b, labels)
            p_b = p_b_dist[y_b]
        else:
            y_b, p_b = None, None

        records.append({
            "idx": i, "gold": s.gold_label,
            "y_a": y_a, "p_a": round(p_a, 6),
            "y_b": y_b, "p_b": (None if p_b is None else round(p_b, 6)),
        })
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            logger.info("  %d/%d（%.2fs/条，剩余约 %.0fs）",
                        i + 1, len(samples), rate, rate * (len(samples) - i - 1))
    return records


def apply_policy(records, tau: float, margin: float):
    """在已收集的四元组上模拟门控+采纳规则，返回预测与统计。"""
    preds, n_trig, n_flip, n_flip_correct, n_flip_wrong = [], 0, 0, 0, 0
    for r in records:
        y_a, p_a, y_b, p_b = r["y_a"], r["p_a"], r["y_b"], r["p_b"]
        flip = False
        if p_a < tau and y_b is not None:
            n_trig += 1
            flip = (y_b != y_a) and (p_b > p_a + margin)
        if flip:
            n_flip += 1
            if y_b == r["gold"]:
                n_flip_correct += 1
            elif y_a == r["gold"]:
                n_flip_wrong += 1
        preds.append(y_b if flip else y_a)
    n = len(records)
    return preds, {
        "trigger_rate": round(n_trig / n, 4),
        "n_trigger": n_trig,
        "n_flip": n_flip,
        "flip_correct": n_flip_correct,
        "flip_wrong": n_flip_wrong,
        "flip_net_correct": n_flip_correct - n_flip_wrong,
    }


def calibrate_on_dev(records, golds, schema):
    """dev 网格标定：W-F1 最大；并列取触发率低（保守、低成本）。"""
    grid = []
    best = None
    for tau in TAUS:
        for margin in MARGINS:
            preds, stats = apply_policy(records, tau, margin)
            m = compute_metrics(preds, golds, schema.labels)
            row = {"tau": tau, "margin": margin, "wf1": round(m["wf1"], 6),
                   "macro_f1": round(m["macro_f1"], 6),
                   "accuracy": round(m["accuracy"], 6), **stats}
            grid.append(row)
            key = (row["wf1"], -row["trigger_rate"], -row["n_flip"])
            if best is None or key > best[0]:
                best = (key, row)
    # proponent-only 内部对照（不触发任何辩论）
    base = compute_metrics([r["y_a"] for r in records], golds, schema.labels)
    return best[1], grid, base


def main() -> int:
    parser = argparse.ArgumentParser(description="选择性辩论 dev 标定 / test 验证")
    parser.add_argument("--mode", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--adapter", default=str(M0_SFT))
    parser.add_argument("--n", type=int, default=0, help="0=全量；>0 时取前 N 条（冒烟用）")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/ablations/dpo_direct_classify.yaml"),
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)

    do_dev = args.mode in ("dev", "both")
    do_test = args.mode in ("test", "both")

    model, tokenizer = load_backbone(cfg.backbone)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="m0_sft")
    model.set_adapter("m0_sft")
    model.eval()
    gen = LLMGenerator(model, tokenizer)
    gen.candidate_labels = list(schema.labels)

    chosen = None
    if do_dev:
        dev = load_split(cfg.data.dataset, cfg.data.raw_path, "dev", cfg.data.history_window)
        if args.n:
            dev = dev[:args.n]
        golds = [s.gold_label for s in dev]
        logger.info("dev 收集四元组 n=%d（全部计算 critic 视图）", len(dev))
        with torch.no_grad():
            dev_records = collect_records(gen, schema, dev, True, None)
        (OUT_DIR / "dev_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in dev_records),
            encoding="utf-8",
        )
        chosen, grid, base = calibrate_on_dev(dev_records, golds, schema)
        calib = {
            "selection_rule": "max dev W-F1; tie → lower trigger rate → fewer flips",
            "chosen": chosen,
            "proponent_only_dev": {
                "wf1": round(base["wf1"], 6),
                "macro_f1": round(base["macro_f1"], 6),
                "accuracy": round(base["accuracy"], 6),
            },
            "grid": grid,
        }
        CALIB_FILE.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "dev 标定完成：τ=%.2f margin=%.2f → W-F1=%.4f（proponent-only %.4f，"
            "触发 %.1f%%，翻转 %d，净改对 %+d）",
            chosen["tau"], chosen["margin"], chosen["wf1"], base["wf1"],
            chosen["trigger_rate"] * 100, chosen["n_flip"], chosen["flip_net_correct"],
        )

    if do_test:
        # 防泄漏：test 参数只允许来自 dev 标定文件
        if chosen is None:
            if not CALIB_FILE.exists():
                raise RuntimeError("test 模式必须先运行 --mode dev 生成标定文件")
            chosen = json.loads(CALIB_FILE.read_text(encoding="utf-8"))["chosen"]
        tau, margin = chosen["tau"], chosen["margin"]
        test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
        if args.n:
            test = test[:args.n]
        golds = [s.gold_label for s in test]
        logger.info("test 验证 n=%d，参数锁定 τ=%.2f margin=%.2f（禁止在 test 调参）",
                    len(test), tau, margin)
        with torch.no_grad():
            test_records = collect_records(gen, schema, test, False, tau)
        (OUT_DIR / "test_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in test_records),
            encoding="utf-8",
        )
        preds, stats = apply_policy(test_records, tau, margin)
        m = compute_metrics(preds, golds, schema.labels)
        base = compute_metrics([r["y_a"] for r in test_records], golds, schema.labels)
        # 触发桶内分析：被门控选中样本上 proponent 与最终决策的正确率
        trig = [r for r in test_records if r["p_a"] < tau]
        trig_golds = [r["gold"] for r in trig]
        summary = {
            "adapter": args.adapter,
            "params_from_dev": {"tau": tau, "margin": margin},
            "n_test": len(test),
            "proponent_only": {
                "wf1": round(base["wf1"], 6), "macro_f1": round(base["macro_f1"], 6),
                "accuracy": round(base["accuracy"], 6),
            },
            "selective_deliberation": {
                "wf1": round(m["wf1"], 6), "macro_f1": round(m["macro_f1"], 6),
                "accuracy": round(m["accuracy"], 6),
                "per_class_f1": {lb: round(m["per_class_f1"].get(lb, 0.0), 4)
                                 for lb in schema.labels},
            },
            "delta_wf1": round(m["wf1"] - base["wf1"], 6),
            "delta_macro": round(m["macro_f1"] - base["macro_f1"], 6),
            **stats,
            "triggered_bucket": {
                "n": len(trig),
                "proponent_acc": round(
                    sum(r["y_a"] == r["gold"] for r in trig) / max(len(trig), 1), 4),
                "final_acc": round(
                    sum(
                        (r["y_b"] if (r["y_b"] is not None and r["y_b"] != r["y_a"]
                                      and r["p_b"] > r["p_a"] + margin) else r["y_a"]) == r["gold"]
                        for r in trig
                    ) / max(len(trig), 1), 4),
            },
        }
        (OUT_DIR / "test_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "test：proponent-only W-F1=%.4f → 选择性辩论 %.4f（%+.4f），"
            "触发 %d/%d，翻转 %d（改对 %d / 改错 %d）",
            base["wf1"], m["wf1"], m["wf1"] - base["wf1"],
            stats["n_trigger"], len(test), stats["n_flip"],
            stats["flip_correct"], stats["flip_wrong"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
