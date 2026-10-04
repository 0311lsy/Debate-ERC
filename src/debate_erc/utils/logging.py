"""统一日志：debate_erc 包根统一挂控制台 handler，子 logger 经 propagate 汇聚；
run 级文件输出经 attach_run_log_file 追加（R2-m13：run 日志落盘到 run 目录）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_NAME = "debate_erc"
_FMT = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def _root_logger() -> logging.Logger:
    """debate_erc 包根 logger：控制台 handler 幂等挂载（子 logger 汇聚到这里）。"""
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.INFO)
    root.propagate = False  # 阻断向 Python root logger 重复输出
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_FMT)
        root.addHandler(console)
    return root


def get_logger(name: str = "debate_erc", log_file: str | Path | None = None) -> logging.Logger:
    """获取包内 logger（控制台输出统一在包根处理，幂等）。

    log_file 传入时在包根追加 run 文件 handler（兼容旧的按名落盘语义）。
    """
    root = _root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if log_file is not None:
        attach_run_log_file(log_file)
    return logger


def attach_run_log_file(log_file: str | Path) -> None:
    """run 级日志落盘（R2-m13）：把 debate_erc 全部日志写入 run 目录文件。

    - 同一路径幂等（重复调用不重复挂）；
    - 同进程多次 run（如 run.py 逐 seed 循环）时移除上一个 run 的文件
      handler（_debate_erc_run 标记），避免旧 run 文件被后续 run 继续写入。
    """
    root = _root_logger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "_debate_erc_run", False):
            root.removeHandler(h)
            h.close()
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_FMT)
    fh._debate_erc_run = True  # type: ignore[attr-defined]
    root.addHandler(fh)
