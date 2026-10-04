"""全局随机种子管理：覆盖 random / numpy / torch / cuda。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """设置全部随机源种子，保证可复现。

    注意：PYTHONHASHSEED 仅对设置后启动的子进程生效——当前解释器进程的
    哈希种子在启动时已固定，无法在运行中修改。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # 仅对子进程生效
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # 无 torch 环境下（纯单测）仅设置 random/numpy
        pass
