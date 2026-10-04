"""辩论流程评估（设计报告 2.2 / V3 6.7 评估指标）：错误共识统计 + 路由统计。

依赖方向：evaluation（与 pipeline 平级）只依赖低层 data/debate/utils，
consensus 统计逻辑复用 debate 层的 summarize_debate（低层依赖，允许）。
（R2-m9：debate.consensus 已就绪，旧的本地上等价回退实现已删除。）
"""

from __future__ import annotations

from collections import Counter

from ..data.schemas import DebateOutcome
from ..debate.consensus import summarize_debate


def _reached_consensus(outcome: DebateOutcome) -> bool:
    """是否达成共识：辩论收敛标记（converged，双方标签一致且置信度达阈）。"""
    return bool(outcome.converged)


def evaluate_debate(
    outcomes: list[DebateOutcome],
    golds: list[str],
    single_labels: list[str] | None = None,
) -> dict:
    """辩论全流程评估：填充错误共识标记后汇总统计。

    步骤：
    1. 对照金标签填充每样本 consensus_but_wrong（双方共识但最终错误）；
    2. 调用 debate.consensus.summarize_debate 汇总
       （debate_accuracy / error_consensus_rate 等键口径以 debate.consensus 为准）；
    3. 若提供 single_labels（单 Agent 基线预测），追加辩论 vs 单 Agent 的
       胜/负/平计数（debate_wins / single_wins / ties）。
    """
    if len(outcomes) != len(golds):
        raise ValueError(f"outcomes 与 golds 长度不一致: {len(outcomes)} vs {len(golds)}")
    if single_labels is not None and len(single_labels) != len(golds):
        raise ValueError(
            f"single_labels 与 golds 长度不一致: {len(single_labels)} vs {len(golds)}"
        )
    # 1) 错误共识填充（评估阶段对照金标签，见 data.schemas.DebateOutcome 注释）
    for outcome, gold in zip(outcomes, golds):
        outcome.consensus_but_wrong = _reached_consensus(outcome) and outcome.final_label != gold
    # 2) 汇总（debate.consensus 为唯一口径）
    stats = dict(summarize_debate(outcomes, golds, single_labels=single_labels))
    # 3) 辩论 vs 单 Agent 对比计数（debate.consensus 未覆盖的部分）
    if single_labels is not None:
        debate_correct = [o.final_label == g for o, g in zip(outcomes, golds)]
        single_correct = [p == g for p, g in zip(single_labels, golds)]
        stats["debate_wins"] = sum(d and not s for d, s in zip(debate_correct, single_correct))
        stats["single_wins"] = sum(s and not d for d, s in zip(debate_correct, single_correct))
        stats["ties"] = sum(d == s for d, s in zip(debate_correct, single_correct))
    return stats


def route_metrics(modes: list[str], correct: list[bool]) -> dict:
    """路由统计：总体/分模式准确率与模式分布。

    参数：
        modes: 每样本的路由模式（如 "PERCEIVE" / "DELIBERATE"）
        correct: 每样本预测是否正确

    返回：
        {"n", "overall_accuracy", "mode_distribution": {mode: 占比},
         "per_mode": {mode: {"n", "accuracy"}}}
    """
    if len(modes) != len(correct):
        raise ValueError(f"modes 与 correct 长度不一致: {len(modes)} vs {len(correct)}")
    n = len(modes)
    if n == 0:
        return {"n": 0, "overall_accuracy": 0.0, "mode_distribution": {}, "per_mode": {}}
    dist = Counter(modes)
    per_mode: dict[str, dict] = {}
    for mode in sorted(dist):
        idx = [i for i, m in enumerate(modes) if m == mode]
        hits = sum(1 for i in idx if correct[i])
        per_mode[mode] = {"n": len(idx), "accuracy": hits / len(idx)}
    overall = sum(1 for c in correct if c) / n
    return {
        "n": n,
        "overall_accuracy": overall,
        "mode_distribution": {mode: count / n for mode, count in sorted(dist.items())},
        "per_mode": per_mode,
    }
