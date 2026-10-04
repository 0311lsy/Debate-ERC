#!/usr/bin/env python
"""异构选择性辩论终审：LLaMA2-7B proponent vs Qwen-1.5B critic。

对同构 self-debate 证伪后的最终检验：两个独立模型族、各自独立 SFT
（数据/提示/超参/LoRA 完全同口径，仅骨干不同）的视角是否具有互补性。

流程与 eval_selective.py 严格一致（dev 标定 τ×margin → test 锁定参数终审），
唯一差异：critic 视图 (y_b, P_B) 由 Qwen-1.5B+LoRA 独立给出（其自带
tokenizer 对标签分词、score_labels 各自归一）。同时报告 critic 独立准确率，
作为互补性解释的能力对照（防止弱模型 critic 的失败被误读）。

用法：
    python scripts/eval_selective_hetero.py --mode both
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

logger = get_logger("debate_erc.scripts.eval_selective_hetero")

OUT_DIR = PROJECT_ROOT / "outputs" / "selective_hetero"
PROP_ADAPTER = PROJECT_ROOT / "outputs/runs/base_sft_s42/adapter_main/main"
CRITIC_ADAPTER = PROJECT_ROOT / "outputs/hetero_critic/qwen15b_sft_s42/adapter_main/main"
CRITIC_BASE = "/home/lsy20252770/InstructERC/LLM_bases/DeepSeek-R1-Distill-Qwen-1.5B"
CALIB_FILE = OUT_DIR / "dev_calibration.json"

TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]
GEN_KW = {"do_sample": False, "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0}


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
    grid, best = [], None
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
    base = compute_metrics([r["y_a"] for r in records], golds, schema.labels)
    return best[1], grid, base


def main() -> int:
    parser = argparse.ArgumentParser(description="异构选择性辩论终审")
    parser.add_argument("--mode", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--prop_adapter", default=str(PROP_ADAPTER))
    parser.add_argument("--critic_base", default=CRITIC_BASE)
    parser.add_argument("--critic_adapter", default=str(CRITIC_ADAPTER))
    parser.add_argument("--tag", default="qwen15b",
                        help="输出子目录名（区分异构 critic 规模/种类）")
    parser.add_argument("--fixed-tau", type=float, default=None,
                        help="跳过 dev 标定，test 直接使用该 τ（参数须来自 pooled-dev 等"
                             " test 不可见来源，禁止在 test 上挑选）")
    parser.add_argument("--fixed-margin", type=float, default=0.05)
    parser.add_argument("--no-critic-hint", action="store_true",
                        help="hint 消融：critic 视图不注入三类错误自检指引"
                             "（其余口径与主实验完全一致）")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/ablations/dpo_direct_classify.yaml"),
    )
    args = parser.parse_args()
    critic_hint = None if args.no_critic_hint else CRITIC_REASONING_HINT
    out_dir = OUT_DIR / args.tag
    calib_file = out_dir / "dev_calibration.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)
    do_dev = args.mode in ("dev", "both")
    do_test = args.mode in ("test", "both")

    logger.info("加载 proponent：LLaMA2-7B + m0_sft LoRA")
    base_model, base_tok = load_backbone(cfg.backbone)
    base_model = PeftModel.from_pretrained(base_model, args.prop_adapter, adapter_name="m0_sft")
    base_model.set_adapter("m0_sft")
    base_model.eval()
    prop = LLMGenerator(base_model, base_tok)
    prop.candidate_labels = list(schema.labels)

    logger.info("加载 critic：%s + %s", args.critic_base, args.critic_adapter)
    crit_model, crit_tok = load_backbone(args.critic_base)
    crit_model = PeftModel.from_pretrained(crit_model, args.critic_adapter, adapter_name="critic")
    crit_model.set_adapter("critic")
    crit_model.eval()
    critic = LLMGenerator(crit_model, crit_tok)
    critic.candidate_labels = list(schema.labels)

    chosen = None
    if do_dev:
        dev = load_split(cfg.data.dataset, cfg.data.raw_path, "dev", cfg.data.history_window)
        golds = [s.gold_label for s in dev]
        logger.info("dev 收集四元组 n=%d（双模型）", len(dev))
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
        calib_file.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("dev：τ=%.2f margin=%.2f W-F1=%.4f（proponent-only %.4f）"
                    " | critic 独立 acc=%.3f proponent acc=%.3f 一致率=%.3f | 净改对 %+d",
                    chosen["tau"], chosen["margin"], chosen["wf1"], base["wf1"],
                    crit_acc, prop_acc, agree, chosen["flip_net_correct"])

    if do_test:
        if args.fixed_tau is not None:
            # 共享阈值口径：参数由外部 dev 来源（如三 seed pooled-dev）给定，
            # 本进程不读取任何 dev_calibration.json，杜绝单 seed 标定漂移。
            tau, margin = args.fixed_tau, args.fixed_margin
        elif chosen is None:
            chosen = json.loads(calib_file.read_text(encoding="utf-8"))["chosen"]
            tau, margin = chosen["tau"], chosen["margin"]
        else:
            tau, margin = chosen["tau"], chosen["margin"]
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
        logger.info("test：%.4f → %.4f（%+.4f）翻转 %d（改对 %d/改错 %d）；critic 独立 W-F1=%.4f",
                    base["wf1"], m["wf1"], m["wf1"] - base["wf1"], stats["n_flip"],
                    stats["flip_correct"], stats["flip_wrong"], crit_only["wf1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
