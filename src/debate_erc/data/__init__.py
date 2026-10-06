"""数据加载统一入口。"""

from __future__ import annotations

from ..utils.label_schema import get_schema
from .iemocap_loader import load_iemocap_split
from .intensity_annotator import annotate_intensity, intensity_label
from .meld_loader import check_official_counts, load_meld_split
from .emorynlp_loader import check_official_counts as check_emorynlp_counts, load_emorynlp_split
from .schemas import AgentOutput, DialogueContext, DialogueSample
from .synthesis import apply_fewshot_synthesis, synthesis_report


def load_split(dataset: str, raw_path: str, split: str, history_window: int = 5):
    """按数据集名分发加载器。"""
    if dataset == "meld":
        return load_meld_split(raw_path, split, history_window)
    if dataset == "iemocap":
        return load_iemocap_split(raw_path, split, history_window)
    if dataset == "emorynlp":
        return load_emorynlp_split(raw_path, split, history_window)
    if dataset == "dailydialog":
        from .dailydialog_loader import load_dailydialog_split
        return load_dailydialog_split(raw_path, split, history_window)
    raise ValueError(f"未知数据集: {dataset}")


__all__ = [
    "load_split", "load_meld_split", "check_official_counts",
    "load_iemocap_split", "annotate_intensity", "intensity_label",
    "DialogueContext", "DialogueSample", "AgentOutput",
    "apply_fewshot_synthesis", "synthesis_report", "get_schema",
]
