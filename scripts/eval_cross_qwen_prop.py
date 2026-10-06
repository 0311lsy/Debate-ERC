#!/usr/bin/env python
"""交叉验证实验：Qwen2.5-7B proponent + LLaMA2-7B critic（角色互换）。

与 eval_selective_hetero.py 严格同口径（冻结 τ=0.65, margin=0.05），
唯一差异：proponent 和 critic 的模型家族互换。
目的：验证"门控机制与具体模型家族无关，只要求独立性+能力对等"。

用法：
    python scripts/eval_cross_qwen_prop.py --seed 42
    python scripts/eval_cross_qwen_prop.py --seed 42 --mode test  # 只跑 test
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

logger = get_logger("debate_erc.scripts.eval_cross_qwen_prop")

OUT_DIR = PROJECT_ROOT / "outputs" / "selective_hetero"
TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]
GEN_KW = {"do_sample": False, "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0}

QWEN_BASE = "/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B"
LLAMA_BASE = "/home/lsy20252770/InstructERC/LLM_bases/LLaMA2"


def collect_records(prop, critic, schema, samples, critic_for_all: bool, gate_tau,
                    critic_hint=CRITIC_REASONING_HINT):
    labels = list(schema.labels)
    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt_a = build_classify_prompt(s.context, schema)
        y_a = parse_label(prop.generate(prompt_a, **GEN_KW), schema)
        p_a = prop.score_labels(prompt_a, labels)[y_a]

        need = critic_for_all or (p_a < gate_tau)
        if need:
            prompt_b = build_classify_prompt(s.context, schema,
                                             reasoning_hint=critic_hint)
            y_b = parse_label(critic.generate(prompt_b, **GEN_KW), schema)
            p_b = critic.score_labels(prompt_b, labels)[y_b]
        else:
            y_b, p_b = None, None
        records.append({"idx": i, "gold": s.gold_label,
                        "y_a": y_a, "p_a": round(p_a, 6),
                        "y_b": y_b, "p_b": (None if p_b is None else round(p_b, 6))})
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            logger.info("  %d/%d（%.2fs/条，剩余约 %.0fs）",
                        i + 1, len(samples), rate, rate * (len(samples) - i - 1))
    return records


def apply_policy(records, tau, margin):
    preds, n_trig, n_flip, nc, nw = [], 0, 0, 0, 0
    for r in records:
        flip = False
        if r["p_a"] < tau and r["y_b"] is not None:
            n_trig += 1
            flip = (r["y_b"] != r["y_a"]) and (r["p_b"] > r["p_a"] + margin)
        if flip:
            n_flip += 1
            nc += int(r["y_b"] == r["gold"])
            nw += int(r["y_a"] == r["gold"] and r["y_b"] != r["gold"])
        preds.append(r["y_b"] if flip else r["y_a"])
    return preds, {"trigger_rate": round(n_trig / len(records), 4), "n_trigger": n_trig,
                   "n_flip": n_flip, "flip_correct": nc, "flip_wrong": nw,
                   "flip_net_correct": nc - nw}


def calibrate(records, golds, schema):
    grid, best = None, None
    for tau in TAUS:
        for margin in MARGINS:
            preds, stats = apply_policy(records, tau, margin)
            m = compute_metrics(preds, golds, schema.labels)
            row = {"tau": tau, "margin": margin, "wf1": round(m["wf1"], 6),
                   "macro_f1": round(m["macro_f1"], 6),
                   "accuracy": round(m["accuracy"], 6), **stats}
            if grid is None:
                grid = []
            grid.append(row)
            key = (row["wf1"], -row["trigger_rate"], -row["n_flip"])
            if best is None or key > best[0]:
                best = (key, row)
    base = compute_metrics([r["y_a"] for r in records], golds, schema.labels)
    return best[1], grid, base


def main() -> int:
    parser = argparse.ArgumentParser(description="交叉验证：Qwen proponent + LLaMA2 critic")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--fixed-tau", type=float, default=0.65)
    parser.add_argument("--fixed-margin", type=float, default=0.05)
    parser.add_argument("--no-critic-hint", action="store_true")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/ablations/dpo_direct_classify.yaml"),
    )
    args = parser.parse_args()
    critic_hint = None if args.no_critic_hint else CRITIC_REASONING_HINT
    seed = args.seed

    out_dir = OUT_DIR / f"cross_qwen_prop_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)

    prop_adapter = str(PROJECT_ROOT / f"outputs/hetero_critic/qwen7b_sft_s{seed}/adapter_main/main")
    critic_adapter = str(PROJECT_ROOT / f"outputs/runs/base_sft_s{seed}/adapter_main/main")

    logger.info("加载 proponent：Qwen2.5-7B + %s", prop_adapter)
    prop_base, prop_tok = load_backbone(QWEN_BASE)
    prop_base = PeftModel.from_pretrained(prop_base, prop_adapter, adapter_name="prop")
    prop_base.set_adapter("prop")
    prop_base.eval()
    prop = LLMGenerator(prop_base, prop_tok)
    prop.candidate_labels = list(schema.labels)

    logger.info("加载 critic：LLaMA2-7B + %s", critic_adapter)
    crit_base, crit_tok = load_backbone(LLAMA_BASE)
    crit_base = PeftModel.from_pretrained(crit_base, critic_adapter, adapter_name="critic")
    crit_base.set_adapter("critic")
    crit_base.eval()
    critic = LLMGenerator(crit_base, crit_tok)
    critic.candidate_labels = list(schema.labels)

    if args.mode in ("dev", "both"):
        dev = load_split(cfg.data.dataset, cfg.data.raw_path, "dev", cfg.data.history_window)
        golds = [s.gold_label for s in dev]
        logger.info("dev 收集四元组 n=%d", len(dev))
        with torch.no_grad():
            recs = collect_records(prop, critic, schema, dev, True, None,
                                   critic_hint=critic_hint)
        (out_dir / "dev_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
        chosen, grid, base = calibrate(recs, golds, schema)
        crit_acc = sum(r["y_b"] == r["gold"] for r in recs) / len(recs)
        prop_acc = sum(r["y_a"] == r["gold"] for r in recs) / len(recs)
        agree = sum(r["y_a"] == r["y_b"] for r in recs) / len(recs)
        calib = {"chosen": chosen,
                 "proponent_only_dev": {"wf1": round(base["wf1"], 6),
                                        "macro_f1": round(base["macro_f1"], 6),
                                        "accuracy": round(base["accuracy"], 6)},
                 "critic_standalone_dev_acc": round(crit_acc, 4),
                 "proponent_dev_acc": round(prop_acc, 4),
                 "agreement": round(agree, 4), "grid": grid}
        (out_dir / "dev_calibration.json").write_text(
            json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("dev：τ=%.2f margin=%.2f W-F1=%.4f | critic acc=%.3f prop acc=%.3f agree=%.3f",
                    chosen["tau"], chosen["margin"], chosen["wf1"],
                    crit_acc, prop_acc, agree)

    if args.mode in ("test", "both"):
        tau, margin = args.fixed_tau, args.fixed_margin
        test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
        golds = [s.gold_label for s in test]
        logger.info("test 终审 n=%d，参数锁定 τ=%.2f margin=%.2f", len(test), tau, margin)
        with torch.no_grad():
            recs = collect_records(prop, critic, schema, test, False, tau,
                                   critic_hint=critic_hint)
        (out_dir / "test_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
        preds, stats = apply_policy(recs, tau, margin)
        m = compute_metrics(preds, golds, schema.labels)
        base = compute_metrics([r["y_a"] for r in recs], golds, schema.labels)
        crit_preds = [r["y_b"] if r["y_b"] is not None else r["y_a"] for r in recs]
        crit_only = compute_metrics(crit_preds, golds, schema.labels)
        trig = [r for r in recs if r["p_a"] < tau]
        summary = {
            "params_from_dev": {"tau": tau, "margin": margin}, "n_test": len(test),
            "seed": seed,
            "proponent_only": {"wf1": round(base["wf1"], 6),
                               "macro_f1": round(base["macro_f1"], 6),
                               "accuracy": round(base["accuracy"], 6)},
            "critic_standalone_all": {"wf1": round(crit_only["wf1"], 6),
                                      "accuracy": round(crit_only["accuracy"], 6)},
            "hetero_deliberation": {"wf1": round(m["wf1"], 6),
                                    "macro_f1": round(m["macro_f1"], 6),
                                    "accuracy": round(m["accuracy"], 6),
                                    "per_class_f1": {lb: round(m["per_class_f1"].get(lb, 0.0), 4)
                                                     for lb in schema.labels}},
            "delta_wf1": round(m["wf1"] - base["wf1"], 6),
            "delta_macro": round(m["macro_f1"] - base["macro_f1"], 6),
            **stats,
            "triggered_bucket": {
                "n": len(trig),
                "proponent_acc": round(sum(r["y_a"] == r["gold"] for r in trig) / max(len(trig), 1), 4),
                "final_acc": round(sum(
                    (r["y_b"] if (r["y_b"] is not None and r["y_b"] != r["y_a"]
                                  and r["p_b"] > r["p_a"] + margin) else r["y_a"]) == r["gold"]
                    for r in trig) / max(len(trig), 1), 4)},
        }
        (out_dir / "test_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("test：%.4f → %.4f（%+.4f）翻转 %d（改对 %d/改错 %d）",
                    base["wf1"], m["wf1"], m["wf1"] - base["wf1"], stats["n_flip"],
                    stats["flip_correct"], stats["flip_wrong"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
