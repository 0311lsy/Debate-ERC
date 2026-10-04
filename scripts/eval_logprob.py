#!/usr/bin/env python
"""Layer 0 测量修复：候选标签首-token logprob 受限分类评估（贪心、可复现）。

审稿式诊断发现原评估链路的两个测量缺陷：
1. do_sample=True/T=0.7 随机采样 → 指标不可复现，组间 <1 点差异落在噪声内；
2. 自报置信度恒为 0.5（模板无 Confidence 字段）→ 路由/收敛/reask 门控全部失真。

本脚本绕过自由生成，直接在「7 个候选标签首 token」上做归一化 softmax：
- 预测 = argmax 标签（确定性，等价贪心但无生成噪声与复读浪费）；
- 置信度 = 该标签在候选集内的概率（校准信号，供分桶/ECE 分析）。

纯前向、批量 left-padding、与训练同 prompt（build_classify_prompt），
全量 2610 条约 1~2 分钟。可对任意 adapter checkpoint 评估，用于同口径横比：
M0-SFT / M1-SFT / M1-DPO 三个 checkpoint 的真实差异。

用法：
    python scripts/eval_logprob.py --checkpoints \
        m0_sft=outputs/runs/base_sft_s42/adapter_main/main \
        m1_sft=outputs/runs/debate_full_.../adapter_main/main \
        m1_dpo=outputs/runs/debate_full_.../dpo/dpo_epoch3/main
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

from debate_erc.agents.prompt_builder import build_classify_prompt  # noqa: E402
from debate_erc.config import load_config  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.models import AdapterManager, load_backbone  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.eval_logprob")

_MAX_LEN = 2048  # 与 SFT left_truncate_with_bos 口径一致


def encode_prompts(prompts: list[str], tokenizer, device):
    """批量编码：left 截断（保尾，与训练口径一致）+ left padding（末位 logits 对齐）。"""
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_LEN,
        padding=True,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_batches(prompts: list[str], labels: list[str], tokenizer,
                  batch_size: int = 32) -> list[dict]:
    """预构造全部 N×L 展开序列的 padded 批次（CPU 张量，跨 checkpoint 复用）。

    每个 checkpoint 的 adapter 只影响前向权重、不改变输入，因此 tokenize /
    拼接 / padding / 位置索引只做一次（原实现每 checkpoint 重复一遍是主要耗时）。
    """
    space_id = tokenizer.encode(" neutral", add_special_tokens=False)[0]
    label_tok: dict[str, list[int]] = {}
    for lb in labels:
        ids = tokenizer.encode(f" {lb}", add_special_tokens=False)
        ids = ids[1:] if ids and ids[0] == space_id else ids
        assert ids, f"标签 {lb} 内容 token 为空"
        label_tok[lb] = ids

    tokenizer.padding_side = "right"
    sequences, prompt_lens, label_lens = [], [], []
    for prompt in prompts:
        pids = tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids
        max_prompt = _MAX_LEN - 4
        if len(pids) > max_prompt:
            pids = [pids[0]] + pids[-(max_prompt - 1):]
        for lb in labels:
            sequences.append(pids + label_tok[lb])
            prompt_lens.append(len(pids))
            label_lens.append(len(label_tok[lb]))

    pad_id = tokenizer.pad_token_id
    batches = []
    for s in range(0, len(sequences), batch_size):
        seqs = sequences[s:s + batch_size]
        B = len(seqs)
        maxlen = max(len(x) for x in seqs)
        maxllen = max(label_lens[s:s + B])
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        tgt = torch.zeros((B, maxllen), dtype=torch.long)
        mask = torch.zeros((B, maxllen), dtype=torch.float32)
        plen_t = torch.empty(B, dtype=torch.long)
        llen_t = torch.empty(B, dtype=torch.long)
        for j, x in enumerate(seqs):
            plen, llen = prompt_lens[s + j], label_lens[s + j]
            input_ids[j, :len(x)] = torch.tensor(x)
            attn[j, :len(x)] = 1
            tgt[j, :llen] = torch.tensor(x[plen:plen + llen])
            mask[j, :llen] = 1.0
            plen_t[j], llen_t[j] = plen, llen
        batches.append({
            "input_ids": input_ids, "attn": attn, "tgt": tgt, "mask": mask,
            "plen": plen_t, "llen": llen_t, "maxlen": maxlen, "maxllen": maxllen,
        })
    return batches


@torch.no_grad()
def score_batches(model, batches: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    """对当前活跃 adapter 跑全部预构造批次，返回 (logsum, logmean) [M]。"""
    sums_all, means_all = [], []
    for b in batches:
        input_ids = b["input_ids"].to(device)
        attn = b["attn"].to(device)
        tgt = b["tgt"].to(device)
        mask = b["mask"].to(device)
        plen_t = b["plen"].to(device)
        llen_t = b["llen"].to(device)
        B = input_ids.shape[0]
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        rows = torch.arange(B, device=device)[:, None]
        pos = (plen_t[:, None] - 1
               + torch.arange(b["maxllen"], device=device)[None, :]
               ).clamp(max=b["maxlen"] - 1)
        gathered = logits[rows, pos]
        logp = torch.log_softmax(gathered, dim=-1)
        tok_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask
        sums = tok_lp.sum(dim=1)
        sums_all.append(sums.cpu())
        means_all.append((sums / llen_t).cpu())
    return torch.cat(sums_all), torch.cat(means_all)


def label_logprobs_batch(model, tokenizer, prompts: list[str], labels: list[str],
                         device, batch_size: int = 32) -> torch.Tensor:
    """兼容旧签名：即时构造批次并打分（单 checkpoint 用；多 checkpoint 请用 build+score）。"""
    batches = build_batches(prompts, labels, tokenizer, batch_size)
    log_sums, log_means = score_batches(model, batches, device)
    n, L = len(prompts), len(labels)
    return log_sums.view(n, L), log_means.view(n, L)


def calibration_buckets(probs: list[float], correct: list[bool]) -> list[dict]:
    """top1 概率十分位桶：置信度均值 vs 准确率（校准诊断）。"""
    buckets = []
    for lo in [i / 10 for i in range(10)]:
        idx = [i for i, p in enumerate(probs) if lo <= p < lo + 0.1]
        if idx:
            buckets.append({
                "range": f"[{lo:.1f},{lo + 0.1:.1f})",
                "n": len(idx),
                "mean_conf": round(sum(probs[i] for i in idx) / len(idx), 3),
                "acc": round(sum(correct[i] for i in idx) / len(idx), 3),
            })
    return buckets


def evaluate_checkpoint(model, adapter_name: str, adapter_dir: str,
                       batches: list[dict], n_samples: int, n_labels: int,
                       schema, golds: list[str], device) -> dict:
    """切换到指定 adapter，对预构造批次打分并计算双口径指标。"""
    labels = list(schema.labels)
    model.set_adapter(adapter_name)
    model.eval()

    t0 = time.time()
    log_sums, log_means = score_batches(model, batches, device)
    log_sums = log_sums.view(n_samples, n_labels)
    log_means = log_means.view(n_samples, n_labels)
    elapsed = time.time() - t0

    def metrics_for(scores: torch.Tensor):
        pred_idx = scores.argmax(dim=-1).tolist()
        preds = [labels[i] for i in pred_idx]
        probs = torch.softmax(scores, dim=-1)
        top = probs.max(dim=-1).values.tolist()
        m = compute_metrics(preds, golds, labels)
        correct = [p == g for p, g in zip(preds, golds)]
        return m, preds, top, correct

    m_sum, preds_sum, top_sum, correct_sum = metrics_for(log_sums)
    m_mean, preds_mean, _, _ = metrics_for(log_means)
    ece = sum(
        abs(b["mean_conf"] - b["acc"]) * b["n"]
        for b in calibration_buckets(top_sum, correct_sum)
    ) / n_samples

    result = {
        "adapter": adapter_dir,
        "n": n_samples,
        "wf1": m_sum["wf1"],
        "macro_f1": m_sum["macro_f1"],
        "accuracy": m_sum["accuracy"],
        "wf1_logmean": m_mean["wf1"],
        "macro_f1_logmean": m_mean["macro_f1"],
        "accuracy_logmean": m_mean["accuracy"],
        "pred_agreement_sum_vs_mean": sum(
            a == b for a, b in zip(preds_sum, preds_mean)
        ) / n_samples,
        "per_class_f1": {lb: round(m_sum["per_class_f1"].get(lb, 0.0), 4)
                         for lb in labels},
        "pred_distribution": {lb: preds_sum.count(lb) for lb in labels},
        "ece": round(ece, 4),
        "mean_top1_conf": round(sum(top_sum) / len(top_sum), 4),
        "calibration": calibration_buckets(top_sum, correct_sum),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info(
        "[%s] logsum: W-F1=%.4f Macro=%.4f Acc=%.4f | logmean: W-F1=%.4f | ECE=%.3f（%.0fs）",
        adapter_name, result["wf1"], result["macro_f1"], result["accuracy"],
        result["wf1_logmean"], ece, elapsed,
    )
    return result, preds_sum, top_sum, log_sums


def main() -> int:
    parser = argparse.ArgumentParser(description="logprob 受限分类评估（贪心可复现）")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/debate_erc_full.yaml"),
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="name=adapter_dir 成对传入，如 m0_sft=/path/to/adapter",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device_obj = None
    schema = get_schema(cfg.data.dataset)
    labels = list(schema.labels)
    test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
    logger.info("test n=%d 标签=%s", len(test), labels)

    ckpts: dict[str, str] = {}
    for item in args.checkpoints:
        name, _, path = item.partition("=")
        if not path:
            logger.error("--checkpoints 格式应为 name=dir，收到 %s", item)
            return 1
        ckpts[name] = path

    model, tokenizer = load_backbone(cfg.backbone)
    # 关键：不能先 add_adapter("main") 再同名 load_adapter——PEFT 对同名加载
    # 静默不替换权重；且不能用 delete+reload（实测切换不生效）。
    # 正确姿势：首个 adapter 用 from_pretrained 命名加载，其余各自独立名字。
    from peft import PeftModel  # noqa: E402

    first_name, first_dir = next(iter(ckpts.items()))
    model = PeftModel.from_pretrained(model, first_dir, adapter_name=first_name)
    device = next(model.parameters()).device
    for name, path in list(ckpts.items())[1:]:
        model.load_adapter(path, adapter_name=name)

    # 输入只构造一次（N×7 展开序列），三 checkpoint 仅切换 adapter 复跑前向
    logger.info("预构造展开批次（N=%d × L=%d）……", len(test), len(labels))
    t_build = time.time()
    prompts = [build_classify_prompt(s.context, schema) for s in test]
    batches = build_batches(prompts, labels, tokenizer, batch_size=args.batch_size)
    logger.info("批次构造完成：%d 批（%.0fs）", len(batches), time.time() - t_build)
    golds = [s.gold_label for s in test]

    all_results = {}
    detail_dump = None
    for name, adapter_dir in ckpts.items():
        if not (Path(adapter_dir) / "adapter_config.json").exists():
            logger.error("adapter 缺失: %s", adapter_dir)
            return 1
        logger.info("[%s] %s", name, adapter_dir)
        result, preds, top_probs, _ = evaluate_checkpoint(
            model, name, adapter_dir, batches, len(test), len(labels),
            schema, golds, device,
        )
        all_results[name] = result
        # 首个 checkpoint 附 gold 行，其余附 pred/top1
        if detail_dump is None:
            detail_dump = [{
                "idx": i, "gold": s.gold_label,
                name + "_pred": preds[i], name + "_p": round(top_probs[i], 4),
            } for i, s in enumerate(test)]
        else:
            for i in range(len(test)):
                detail_dump[i][name + "_pred"] = preds[i]
                detail_dump[i][name + "_p"] = round(top_probs[i], 4)
        torch.cuda.empty_cache()

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "outputs" / "eval_logprob_comparison.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    detail_path = out_path.with_name(out_path.stem + "_details.jsonl")
    with open(detail_path, "w", encoding="utf-8") as fh:
        for row in detail_dump:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 简表
    print("\n=== logprob 受限分类（贪心，全量 %d）===" % len(test))
    print("%-10s %14s %14s %7s %7s" % ("ckpt", "W-F1(sum/mean)", "Macro(sum/mean)", "Acc", "ECE"))
    for name, r in all_results.items():
        print("%-10s %6.4f/%-6.4f %6.4f/%-6.4f %7.4f %7.3f"
              % (name, r["wf1"], r["wf1_logmean"], r["macro_f1"],
                 r["macro_f1_logmean"], r["accuracy"], r["ece"]))
    print("\n明细:", detail_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
