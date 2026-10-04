"""评估指标（V3 4.4 评估协议 / 8.x 错误分析口径）：纯函数，不依赖模型。

- compute_metrics：wf1 / macro_f1 / accuracy / per-class F1 与 P/R
- error_pair_counts：gold→pred 错误对计数（降序，对齐 V3 错误分析口径）
- format_metrics：人类可读表格字符串
"""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(preds: list[str], golds: list[str], labels: tuple) -> dict:
    """分类指标（sklearn，zero_division=0）。

    参数：
        preds: 预测标签序列（已经过 post_process.parse_label 合法化）
        golds: 金标签序列
        labels: 标签表（如 MELD_SCHEMA.labels），per-class 统计按此顺序

    返回：
        {"wf1", "macro_f1", "accuracy",
         "per_class_f1": {label: f1},
         "per_class_pr": {label: {"precision": p, "recall": r}}}
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds 与 golds 长度不一致: {len(preds)} vs {len(golds)}")
    label_list = list(labels)
    if not preds:
        return {
            "wf1": 0.0, "macro_f1": 0.0, "accuracy": 0.0,
            "per_class_f1": {lb: 0.0 for lb in label_list},
            "per_class_pr": {lb: {"precision": 0.0, "recall": 0.0} for lb in label_list},
        }
    wf1 = f1_score(golds, preds, labels=label_list, average="weighted", zero_division=0)
    macro_f1 = f1_score(golds, preds, labels=label_list, average="macro", zero_division=0)
    accuracy = accuracy_score(golds, preds)
    per_f1 = f1_score(golds, preds, labels=label_list, average=None, zero_division=0)
    pr, rc, _f1s, _sup = precision_recall_fscore_support(
        golds, preds, labels=label_list, zero_division=0
    )
    return {
        "wf1": float(wf1),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_f1": {lb: float(v) for lb, v in zip(label_list, per_f1)},
        "per_class_pr": {
            lb: {"precision": float(p), "recall": float(r)}
            for lb, p, r in zip(label_list, pr, rc)
        },
    }


def error_pair_counts(
    preds: list[str], golds: list[str], top_k: int = 15
) -> list[tuple[str, str, int]]:
    """错误对统计（V3 错误分析口径）：gold→pred 计数降序。

    返回 [(gold, pred, count), ...]，按 count 降序、错误对字典序打破平局，
    截取前 top_k（如 [("sadness", "neutral", 87), ...]）。
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds 与 golds 长度不一致: {len(preds)} vs {len(golds)}")
    counts = Counter((g, p) for g, p in zip(golds, preds) if g != p)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(g, p, c) for (g, p), c in ranked[:top_k]]


def format_metrics(metrics: dict) -> str:
    """把 compute_metrics 的输出渲染为可读表格字符串。"""
    lines = [f"{'Metric':<12}{'Value':>10}", "-" * 22]
    for key in ("wf1", "macro_f1", "accuracy"):
        lines.append(f"{key:<12}{float(metrics.get(key, float('nan'))):>10.4f}")
    per_f1: dict = metrics.get("per_class_f1", {})
    if per_f1:
        lines += ["", f"{'Class':<12}{'F1':>10}{'Precision':>11}{'Recall':>9}", "-" * 42]
        for lb, f1v in per_f1.items():
            pr = metrics.get("per_class_pr", {}).get(lb, {})
            precision = float(pr.get("precision", float("nan")))
            recall = float(pr.get("recall", float("nan")))
            lines.append(f"{lb:<12}{f1v:>10.4f}{precision:>11.4f}{recall:>9.4f}")
    return "\n".join(lines)
