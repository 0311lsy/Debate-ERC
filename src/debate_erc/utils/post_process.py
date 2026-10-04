"""post_process：LLaMA2 无 chat template，输出统一按首行截断 + 标签合法性映射。

复现教训（V3 八、设计报告八）：不依赖模型自行输出 <|im_end|> 停止。
"""

from __future__ import annotations

from .label_schema import LabelSchema, extract_label_token, map_illegal_label

STOP_MARKS = ("<|im_end|>", "<|endoftext|>", "</s>", "\n")


def first_line(text: str) -> str:
    """截断到首个换行 / 停止符。"""
    text = text or ""
    cut = len(text)
    for mark in STOP_MARKS:
        idx = text.find(mark)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def parse_label(text: str, schema: LabelSchema) -> str:
    """完整解析链：首行截断 → 提取标签词 → 非法映射。"""
    line = first_line(text)
    token = extract_label_token(line)
    if token is None:
        return schema.labels[0]  # 空输出的确定性兜底
    return map_illegal_label(token, schema)
