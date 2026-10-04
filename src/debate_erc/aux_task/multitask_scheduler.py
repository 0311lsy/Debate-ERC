"""多任务交替调度器（V3 6.8.3 / 设计报告 2.1）：主 1.0 / 辅 0.3，每 switch_steps 步交替。

实现口径（外层交替，规避 trl 闭包 OOM 教训，见 sft_trainer 模块注释）：
- 主任务与每个辅助任务各绑定一个 SFTTrainer + 任务特定 adapter（AdapterManager）；
- 每 switch_steps 个优化器步切换一次任务：activate("main") 训 main，
  再逐个 activate(aux) 训 aux；
- 辅任务权重 loss_weight（0.3）以步数配比体现：辅助任务总预算 =
  weight × total_main（S5 契约），每个 chunk 内按剩余预算分摊；
- 每个 trainer 的 AdamW 跨 chunk 复用（train(optimizer=...)），避免每 100 步重置动量；
- activate 同步 requires_grad（AdapterManager._sync_trainability），
  保证任一时刻只有活跃 adapter 的参数收到梯度更新。

总步数预算 total_main = min(steps_per_epoch × epochs, max_steps)（max_steps>0 时，
S5 契约；与单任务训练时长对齐，辅助任务的额外步数是训练开销，计入总时长
但不减少主任务步数）。各任务的 dataset 在训练开始前构建一次，
经 train(..., dataset=...) 跨 chunk 复用，避免重复 tokenize 全量样本。
"""

from __future__ import annotations

import math

import torch

from ..models.adapter_manager import AdapterManager
from ..training.sft_trainer import SFTTrainer
from ..utils.logging import get_logger

MAIN_ADAPTER = "main"


class MultitaskScheduler:
    """主任务 + 辅助任务的交替训练调度器。"""

    def __init__(
        self,
        main_trainer: SFTTrainer,
        aux_trainers: dict[str, tuple[SFTTrainer, float]],
        adapter_manager: AdapterManager,
        switch_steps: int = 100,
        logger=None,
    ):
        """
        参数：
            main_trainer: 主任务 SFTTrainer（绑定 "main" adapter）
            aux_trainers: {adapter_name: (SFTTrainer, loss_weight)}
                如 {"intensity": (trainer, 0.3)}
            adapter_manager: adapter 切换管理器（须已注册 main 与各 aux adapter）
            switch_steps: 每次交替的主任务优化器步数（V3 6.8.3 默认 100）
        """
        self.main_trainer = main_trainer
        self.aux_trainers = dict(aux_trainers)
        self.adapter_manager = adapter_manager
        self.switch_steps = max(1, int(switch_steps))
        self.logger = logger or get_logger("debate_erc.aux_task.scheduler")
        self._optimizers: dict[str, torch.optim.Optimizer] = {}

    # ------------------------------------------------------------------
    # 优化器（跨 chunk 复用）
    # ------------------------------------------------------------------

    def _optimizer(self, key: str, trainer: SFTTrainer) -> torch.optim.Optimizer:
        """惰性创建并缓存各任务的 AdamW；须在对应 adapter 激活后首次调用。"""
        if key not in self._optimizers:
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError(f"任务 {key} 无可训练参数（adapter 未激活？）")
            self._optimizers[key] = torch.optim.AdamW(params, lr=trainer.config.lr)
        return self._optimizers[key]

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(
        self,
        main_samples: list,
        aux_samples: dict[str, list],
        eval_fn=None,
    ) -> dict:
        """交替训练至主任务步数预算耗尽。

        参数：
            main_samples: 主任务 DialogueSample 列表
            aux_samples: {adapter_name: 辅助样本列表}，键须 ⊆ aux_trainers
            eval_fn: 主任务每轮 chunk 结束后的可选回调（与 SFTTrainer.eval_fn 同义）

        返回：{"main_steps", "aux_steps": {name: steps}, "final_loss", "eval_history"}

        S5 契约：
        - 各任务 dataset 只构建一次（trainer.build_dataset），
          跨 chunk 经 train(..., dataset=...) 复用；
        - total_main = min(steps_per_epoch × epochs, max_steps)（max_steps>0 时）；
        - 辅助任务总预算 = loss_weight × total_main，每个 chunk 内按剩余预算分摊
          （chunk 预算 = min(round(chunk × weight), 剩余预算)，≤0 时跳过该任务；
          weight ≤ 0 → 总预算 0，任务完全不训练，R2-m5 修复）。
        """
        missing = set(self.aux_trainers) - set(aux_samples)
        if missing:
            raise ValueError(f"辅助任务缺少样本: {sorted(missing)}")

        cfg = self.main_trainer.config
        steps_per_epoch = math.ceil(
            len(main_samples) / cfg.batch_size / cfg.grad_accum
        ) if main_samples else 0
        total_main = steps_per_epoch * cfg.epochs
        if cfg.max_steps > 0:
            total_main = min(total_main, cfg.max_steps)
        aux_steps: dict[str, int] = {name: 0 for name in self.aux_trainers}
        summary: dict = {
            "main_steps": 0, "aux_steps": aux_steps,
            "final_loss": float("nan"), "eval_history": [],
        }
        if total_main <= 0:
            self.logger.warning("[mtl] 主任务步数预算为 0（空样本？），跳过训练")
            return summary

        # ---- 各任务 dataset 只构建一次，跨 chunk 复用（S5 契约）----
        main_dataset = self.main_trainer.build_dataset(main_samples)
        aux_datasets = {
            name: trainer.build_dataset(aux_samples[name])
            for name, (trainer, _w) in self.aux_trainers.items()
        }
        # ---- 辅助任务总预算 = weight × total_main（S5 契约）----
        # weight ≤ 0 → 预算 0（任务跳过）：不再用 max(1, ...) 兜底，
        # 否则 weight=0 时每 chunk 仍会训 1 步（R2-m5）
        aux_remaining: dict[str, int] = {
            name: max(0, round(float(weight) * total_main))
            for name, (_t, weight) in self.aux_trainers.items()
        }

        done = 0
        cycle = 0
        while done < total_main:
            chunk = min(self.switch_steps, total_main - done)
            # ---- 主任务 chunk ----
            self.adapter_manager.activate(MAIN_ADAPTER)
            res = self.main_trainer.train(
                main_samples,
                eval_fn=eval_fn if cycle == 0 else None,  # eval 只挂首轮，避免过频生成
                steps_budget=chunk,
                optimizer=self._optimizer(MAIN_ADAPTER, self.main_trainer),
                dataset=main_dataset,
            )
            done += res["steps"]
            summary["final_loss"] = res["final_loss"]
            summary["eval_history"].extend(res.get("eval_history", []))
            # ---- 辅任务 chunk（预算 = min(chunk × weight, 剩余总预算)，≤0 跳过）----
            for name, (trainer, weight) in self.aux_trainers.items():
                if not aux_samples[name] or aux_remaining[name] <= 0:
                    continue
                aux_budget = min(
                    round(chunk * float(weight)), aux_remaining[name]
                )
                if aux_budget <= 0:
                    continue  # 本 chunk 配额 < 1 步：不训练也不消耗剩余预算
                self.adapter_manager.activate(name)
                aux_res = trainer.train(
                    aux_samples[name],
                    steps_budget=aux_budget,
                    optimizer=self._optimizer(name, trainer),
                    dataset=aux_datasets[name],
                )
                aux_steps[name] += aux_res["steps"]
                aux_remaining[name] -= aux_res["steps"]
            cycle += 1
            self.logger.info(
                "[mtl] cycle %d: main %d/%d steps, aux %s",
                cycle, done, total_main,
                {k: v for k, v in aux_steps.items()},
            )

        # 收尾：回到 main adapter（推理/保存口径）
        self.adapter_manager.activate(MAIN_ADAPTER)
        summary["main_steps"] = done
        return summary
