"""评估层（Layer 6）：分类指标、辩论统计、显著性检验（V3 4.4）。"""

from __future__ import annotations

from .debate_metrics import evaluate_debate, route_metrics
from .metrics import compute_metrics, error_pair_counts, format_metrics
from .significance import (
    bootstrap_ci,
    cohens_d,
    compare_systems,
    paired_ttest,
    wf1,
)

__all__ = [
    "compute_metrics",
    "error_pair_counts",
    "format_metrics",
    "evaluate_debate",
    "route_metrics",
    "bootstrap_ci",
    "paired_ttest",
    "cohens_d",
    "compare_systems",
    "wf1",
]
