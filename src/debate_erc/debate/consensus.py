"""共识统计（V3 6.7 评估指标）：纯统计函数，不依赖模型。

指标口径：
- error_consensus_rate：双方一致（converged=True）但 final_label != gold 的比例（分母=全体样本）；
- convergence_rate：converged=True 的比例；
- correction_rate：单 Agent（proponent 第一轮）错误而辩论后正确的比例（分母=单 Agent 错误样本）；
- introduced_error_rate：单 Agent 正确而辩论后错误的比例（分母=单 Agent 正确样本）；
- reask_trigger_rate / reask_correction_rate：触发兜底的比例 / 触发兜底的样本中
  「reask 前错误且 reask 后正确」的比例（reask_pre_label 缺失时回退旧口径）。
"""

from __future__ import annotations

from ..data.schemas import DebateOutcome


def _check_same_length(*lists) -> None:
    lengths = {len(x) for x in lists}
    if len(lengths) > 1:
        raise ValueError(f"输入列表长度不一致: {sorted(lengths)}")


def compute_error_consensus_rate(outcomes: list[DebateOutcome], golds: list[str]) -> float:
    """错误共识率：converged=True 且 final_label != gold 的样本占全体比例。"""
    _check_same_length(outcomes, golds)
    if not outcomes:
        return 0.0
    n = sum(1 for o, g in zip(outcomes, golds) if o.converged and o.final_label != g)
    return n / len(outcomes)


def compute_convergence_rate(outcomes: list[DebateOutcome]) -> float:
    """辩论收敛率：converged=True 的样本占全体比例。"""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.converged) / len(outcomes)


def compute_correction_rate(
    single_labels: list[str], outcomes: list[DebateOutcome], golds: list[str]
) -> float:
    """修正率：单 Agent 错误而辩论后正确的比例（分母=单 Agent 错误样本数）。"""
    _check_same_length(single_labels, outcomes, golds)
    wrong = [(o, g) for s, o, g in zip(single_labels, outcomes, golds) if s != g]
    if not wrong:
        return 0.0
    corrected = sum(1 for o, g in wrong if o.final_label == g)
    return corrected / len(wrong)


def compute_introduced_error_rate(
    single_labels: list[str], outcomes: list[DebateOutcome], golds: list[str]
) -> float:
    """引入错误率：单 Agent 正确而辩论后错误的比例（分母=单 Agent 正确样本数）。"""
    _check_same_length(single_labels, outcomes, golds)
    right = [(o, g) for s, o, g in zip(single_labels, outcomes, golds) if s == g]
    if not right:
        return 0.0
    introduced = sum(1 for o, g in right if o.final_label != g)
    return introduced / len(right)


def compute_reask_trigger_rate(outcomes: list[DebateOutcome]) -> float:
    """Re-ask 兜底触发率：reask_triggered=True 的样本占全体比例。"""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.reask_triggered) / len(outcomes)


def _is_reask_corrected(outcome: DebateOutcome, gold: str) -> bool:
    """单样本是否算 reask 修正：reask 前错误且 reask 后正确。

    reask_pre_label 为 None（旧数据未记录）时回退旧口径（final_label == gold）。
    """
    if outcome.reask_pre_label is None:
        return outcome.final_label == gold
    return outcome.reask_pre_label != gold and outcome.final_label == gold


def compute_reask_correction_rate(outcomes: list[DebateOutcome], golds: list[str]) -> float:
    """Re-ask 修正率：触发兜底的样本中「reask 前错误且 reask 后正确」的比例。"""
    _check_same_length(outcomes, golds)
    triggered = [(o, g) for o, g in zip(outcomes, golds) if o.reask_triggered]
    if not triggered:
        return 0.0
    return sum(1 for o, g in triggered if _is_reask_corrected(o, g)) / len(triggered)


def summarize_debate(
    outcomes: list[DebateOutcome],
    golds: list[str],
    single_labels: list[str] | None = None,
) -> dict:
    """汇总全部共识指标（V3 6.7）。

    single_labels 提供时（单 Agent 基线预测），额外给出修正率 / 引入错误率与
    单 Agent 准确率；所有 rate 字段在分母为 0 时取 0.0。
    """
    _check_same_length(outcomes, golds)
    if single_labels is not None:
        _check_same_length(single_labels, outcomes)

    n = len(outcomes)
    n_converged = sum(1 for o in outcomes if o.converged)
    n_error_consensus = sum(
        1 for o, g in zip(outcomes, golds) if o.converged and o.final_label != g
    )
    n_reask = sum(1 for o in outcomes if o.reask_triggered)
    n_reask_corrected = sum(
        1 for o, g in zip(outcomes, golds) if o.reask_triggered and _is_reask_corrected(o, g)
    )
    n_debate_correct = sum(1 for o, g in zip(outcomes, golds) if o.final_label == g)

    report: dict = {
        "n": n,
        "convergence_rate": n_converged / n if n else 0.0,
        "error_consensus_rate": n_error_consensus / n if n else 0.0,
        "reask_trigger_rate": n_reask / n if n else 0.0,
        "reask_correction_rate": n_reask_corrected / n_reask if n_reask else 0.0,
        "debate_accuracy": n_debate_correct / n if n else 0.0,
        "n_converged": n_converged,
        "n_error_consensus": n_error_consensus,
        "n_reask_triggered": n_reask,
        "n_reask_corrected": n_reask_corrected,
    }

    if single_labels is not None:
        n_single_wrong = sum(1 for s, g in zip(single_labels, golds) if s != g)
        n_single_right = n - n_single_wrong
        n_corrected = sum(
            1
            for s, o, g in zip(single_labels, outcomes, golds)
            if s != g and o.final_label == g
        )
        n_introduced = sum(
            1
            for s, o, g in zip(single_labels, outcomes, golds)
            if s == g and o.final_label != g
        )
        report.update(
            {
                "single_accuracy": n_single_right / n if n else 0.0,
                "correction_rate": n_corrected / n_single_wrong if n_single_wrong else 0.0,
                "introduced_error_rate": (
                    n_introduced / n_single_right if n_single_right else 0.0
                ),
                "n_corrected": n_corrected,
                "n_introduced_error": n_introduced,
            }
        )
    return report
