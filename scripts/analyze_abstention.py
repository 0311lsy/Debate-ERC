#!/usr/bin/env python
"""Layer 1 备选贡献线：选择性预测（abstention / selective prediction）分析。

Layer 0/1 证据：标签序列概率作为"困难度信号"分桶校准单调，且门控阈值跨
dev/test 稳定（19.7% vs 19.6%）；同构辩论无法利用该信号（改对/改错 48:49）。
本脚本评估信号的标准 selective-prediction 价值（Geifman & El-Yaniv 2017）：

- 按置信度降序作答、对低置信尾部弃权 → 覆盖度-风险曲线与 AURC；
- 对照：随机弃权（AURC=整体错误率）、oracle 弃权（错误样本最先被弃权）；
- 固定风险预算（5%/10%）下可达覆盖率；
- 弃权 5/10/20/30% 最低置信样本后作答集的 W-F1/Macro/Acc；
- 对比 m0_sft 与 m1_dpo 两套概率的门控质量（DPO 校准恶化是否损害弃权决策）。

纯数据分析（读 Layer 0 已落盘明细，零模型加载）。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import compute_metrics  # noqa: E402
from debate_erc.utils.label_schema import MELD_SCHEMA  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402

logger = get_logger("debate_erc.scripts.abstention")
OUT = PROJECT_ROOT / "outputs/abstention"

LABELS = list(MELD_SCHEMA.labels)
ABSTAIN_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40]
RISK_BUDGETS = [0.05, 0.10]


def load_rows():
    """合并 logprob 概率明细与贪心预测明细（两脚本同序 load_split）。"""
    lp = {}
    with open(PROJECT_ROOT / "outputs/eval_logprob_layer0_details.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            lp[r["idx"]] = r
    rows = {idx: {"idx": idx, "gold": r["gold"],
                  "m0_sft_p": r["m0_sft_p"], "m1_dpo_p": r["m1_dpo_p"]}
            for idx, r in lp.items()}
    for ck in ("m0_sft", "m1_dpo"):
        with open(PROJECT_ROOT / f"outputs/eval_greedy.{ck}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows[r["idx"]][f"{ck}_pred"] = r["pred"]
    return [rows[i] for i in sorted(rows)]


def risk_coverage_curve(rows, pred_key, conf_key):
    """按置信度降序返回 (coverages, risks, ordered_correct)。"""
    ordered = sorted(rows, key=lambda r: r[conf_key], reverse=True)
    n = len(ordered)
    correct = [int(r[pred_key] == r["gold"]) for r in ordered]
    cum_err, covs, risks = 0, [], []
    for i, c in enumerate(correct, start=1):
        cum_err += 1 - c
        covs.append(i / n)
        risks.append(cum_err / i)
    return covs, risks, correct


def aurc(covs, risks) -> float:
    """梯形积分 ∫ risk(c) dc（c∈(0,1]）。"""
    area = 0.0
    for i in range(1, len(covs)):
        area += (covs[i] - covs[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return area


def oracle_aurc(rows, pred_key) -> float:
    n = len(rows)
    n_err = sum(r[pred_key] != r["gold"] for r in rows)
    # 理想排序：n-n_err 个正确样本风险恒 0，之后错误样本进入使风险线性升至错误率
    covs, risks, area, cum_err = [], [], 0.0, 0
    for i in range(1, n + 1):
        if i > n - n_err:
            cum_err += 1
        covs.append(i / n)
        risks.append(cum_err / i)
    return aurc(covs, risks)


def random_aurc(rows, pred_key, seed=2026, trials=20) -> float:
    n = len(rows)
    correct = [int(r[pred_key] == r["gold"]) for r in rows]
    rng = random.Random(seed)
    vals = []
    for _ in range(trials):
        perm = correct[:]
        rng.shuffle(perm)
        cum_err, covs, risks = 0, [], []
        for i, c in enumerate(perm, start=1):
            cum_err += 1 - c
            covs.append(i / n)
            risks.append(cum_err / i)
        vals.append(aurc(covs, risks))
    return sum(vals) / len(vals)


def coverage_at_risk(covs, risks, alpha: float) -> float:
    """风险 ≤ alpha 时的最大覆盖率（曲线单调不减，取最后一个满足点）。"""
    best = 0.0
    for c, r in zip(covs, risks):
        if r <= alpha:
            best = c
    return best


def abstain_table(rows, pred_key, conf_key):
    """弃权最低置信尾部后，作答子集的性能（含按比例随机弃权对照）。"""
    n = len(rows)
    out = {}
    for frac in ABSTAIN_FRACTIONS:
        keep_n = round(n * (1 - frac))
        ordered = sorted(rows, key=lambda r: r[conf_key], reverse=True)[:keep_n]
        m = compute_metrics([r[pred_key] for r in ordered],
                            [r["gold"] for r in ordered], LABELS)
        rng = random.Random(2026)
        rand_rows = rows[:]
        rng.shuffle(rand_rows)
        rand_rows = rand_rows[:keep_n]
        mr = compute_metrics([r[pred_key] for r in rand_rows],
                             [r["gold"] for r in rand_rows], LABELS)
        out[f"abstain_{int(frac*100)}pct"] = {
            "coverage": round(keep_n / n, 4),
            "selective_wf1": round(m["wf1"], 4),
            "selective_macro": round(m["macro_f1"], 4),
            "selective_acc": round(m["accuracy"], 4),
            "random_abstain_wf1": round(mr["wf1"], 4),
            "random_abstain_acc": round(mr["accuracy"], 4),
        }
    return out


def main() -> int:
    rows = load_rows()
    logger.info("合并明细 n=%d", len(rows))
    report = {"n": len(rows), "checkpoints": {}}

    for ck in ("m0_sft", "m1_dpo"):
        pred_key, conf_key = f"{ck}_pred", f"{ck}_p"
        base = compute_metrics([r[pred_key] for r in rows],
                               [r["gold"] for r in rows], LABELS)
        covs, risks, _ = risk_coverage_curve(rows, pred_key, conf_key)
        entry = {
            "full_metrics": {
                "wf1": round(base["wf1"], 4), "macro_f1": round(base["macro_f1"], 4),
                "accuracy": round(base["accuracy"], 4),
                "error_rate": round(1 - base["accuracy"], 4),
            },
            "aurc_confidence": round(aurc(covs, risks), 6),
            "aurc_random": round(random_aurc(rows, pred_key), 6),
            "aurc_oracle": round(oracle_aurc(rows, pred_key), 6),
            "coverage_at_risk": {
                f"risk<={a}": round(coverage_at_risk(covs, risks, a), 4)
                for a in RISK_BUDGETS
            },
            "abstention": abstain_table(rows, pred_key, conf_key),
        }
        # excess AURC：相对 oracle 的可避免风险面积（越小越好）
        entry["excess_aurc_vs_oracle"] = round(
            entry["aurc_confidence"] - entry["aurc_oracle"], 6)
        report["checkpoints"][ck] = entry
        logger.info(
            "[%s] AURC 置信=%.4f 随机=%.4f oracle=%.4f | 风险≤5%%覆盖 %.1f%%、≤10%%覆盖 %.1f%%",
            ck, entry["aurc_confidence"], entry["aurc_random"], entry["aurc_oracle"],
            entry["coverage_at_risk"]["risk<=0.05"] * 100,
            entry["coverage_at_risk"]["risk<=0.1"] * 100,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selective_prediction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 选择性预测（test n=%d）===" % len(rows))
    for ck, e in report["checkpoints"].items():
        print(f"\n[{ck}] 全量 W-F1={e['full_metrics']['wf1']} Acc={e['full_metrics']['accuracy']}")
        print(f"  AURC: 置信排序 {e['aurc_confidence']:.4f} | 随机 {e['aurc_random']:.4f} "
              f"| oracle {e['aurc_oracle']:.4f}")
        for k, v in e["abstention"].items():
            print(f"  {k:14s} 覆盖 {v['coverage']:.2f} → W-F1 {v['selective_wf1']:.4f}"
                  f"（随机弃权 {v['random_abstain_wf1']:.4f}）Acc {v['selective_acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
