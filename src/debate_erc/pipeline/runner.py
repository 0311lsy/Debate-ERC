"""ExperimentRunner（设计报告 2.1 #11）：单次 run 的全生命周期编排。

流程（M0→M1 闭环）：
  seed → 建目录/run_meta → 加载数据（+定向合成/CB 权重）→ 主干+LoRA adapter
  → SFT（辅助任务经 MultitaskScheduler 交替；focal/CB 权重经 SFTTrainer）
  → 推理运行时装配（build_runtime + PipelineAssembler）
  → dev 辩论推理 → 金标签偏好对 → DPO
  → test 推理 → 双口径指标（全样本 / 仅真实子集，红线 #6）
  → 辩论过程指标 + 路由统计 / Oracle → results.json / predictions.jsonl 落盘

红线落地：
- #2 输出目录禁 /tmp（config.validate_config + 此处 resolve 到项目持久目录）
- #3 run_meta.json 含 config 快照 + git hash + torch/cuda 版本，任意 run 可复现
- #4 显存峰值落盘（<45GB）
- #5 推理输出经 LabelStage 合法化（首行截断 + 非法映射在 Agent 解析层完成）
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ..agents import DirectClassifyAgent
from ..config import ExperimentConfig, LoRAConfig, make_run_id
from ..data import load_split
from ..data.schemas import DebateOutcome, DialogueSample
from ..evaluation import compute_metrics, error_pair_counts, evaluate_debate, route_metrics
from ..fewshot import class_balanced_weights
from ..models import AdapterManager, LLMGenerator, load_backbone
from ..routing import OracleUpperBound
from ..routing.difficulty_router import RouteMode
from ..training import DPOTrainer, SFTTrainer, build_pairs
from ..utils.label_schema import get_schema
from ..utils.logging import attach_run_log_file, get_logger
from ..utils.seed import set_global_seed
from .assembler import (
    PipelineAssembler,
    Runtime,
    build_generation_kwargs,
    build_runtime,
)
from .stages import run_pipeline

logger = get_logger("debate_erc.pipeline.runner")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _reset_peak_memory_stats() -> None:
    """S6：每 seed run 开头重置 CUDA 峰值显存统计（红线 #4 口径正确）。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_memory_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


class ExperimentRunner:
    """单配置 × 单 seed 的端到端实验执行器。"""

    def __init__(self, config: ExperimentConfig, seed: int,
                 project_root: Path | None = None):
        self.config = config
        self.seed = seed
        self.project_root = (project_root or Path(__file__).resolve().parents[3])
        self.run_id = make_run_id(config, seed)
        out = Path(config.output_dir)
        self.run_dir = out if out.is_absolute() else (self.project_root / out)
        self.run_dir = self.run_dir / self.run_id
        self.schema = get_schema(config.data.dataset)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.config
        set_global_seed(self.seed)
        _reset_peak_memory_stats()  # S6：每 seed run 重置峰值显存统计
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # R2-m13：run 级日志落盘（run.log），长跑日志不再随终端丢失
        attach_run_log_file(self.run_dir / "run.log")
        self._write_run_meta()
        t0 = time.time()
        logger.info("[run] %s 启动，输出目录 %s", self.run_id, self.run_dir)

        # ---- 1) 数据 ----
        train = load_split(cfg.data.dataset, cfg.data.raw_path, "train", cfg.data.history_window)
        dev = load_split(cfg.data.dataset, cfg.data.raw_path, "dev", cfg.data.history_window)
        test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
        synthesis_stats = None
        if cfg.fewshot.synthesis_enabled:
            from ..data import apply_fewshot_synthesis, synthesis_report

            before = list(train)
            train = apply_fewshot_synthesis(train, cfg.fewshot.synthesis_ratio, self.seed)
            synthesis_stats = synthesis_report(before, train)
            logger.info("[run] 定向合成: %s", synthesis_stats)
        if cfg.smoke_samples > 0:
            train = train[: cfg.smoke_samples]
            dev = dev[: cfg.smoke_samples]
            test = test[: cfg.smoke_samples]
            logger.info("[run] smoke 模式：每 split 截取 %d 条", cfg.smoke_samples)

        # ---- 2) 模型与 adapter ----
        model, tokenizer = load_backbone(cfg.backbone)
        adapter_manager: AdapterManager | None = None
        train_summaries: dict = {}
        if cfg.lora.enabled:
            adapter_manager = AdapterManager(model, tokenizer)
            adapter_manager.add("main", cfg.lora)
            # S3（C1）：add() 内部 get_peft_model 包装后重绑定局部 model，
            # 保证 SFTTrainer / DPOTrainer / LLMGenerator / build_runtime
            # 全部持有同一 PeftModel 引用（否则后续训练/推理用的是裸主干）
            model = adapter_manager.model
            trainable, total, ratio = adapter_manager.trainable_params()
            logger.info(
                "[run] main LoRA 可训练参数 %.1fM / %.1fB (%.2f%%)",
                trainable / 1e6, total / 1e9, ratio * 100,
            )

        # 生成器绑定重绑定后的 PeftModel；SFT 期轻量评估（S6）也依赖它
        generator = LLMGenerator(model, tokenizer)
        generator.candidate_labels = list(self.schema.labels)

        # ---- 3) SFT（含辅助任务 / focal / CB 权重）----
        if cfg.lora.enabled and cfg.training.sft_enabled:
            # S6：eval_every_steps > 0 时构造轻量评估回调
            # （direct agent 对 dev 前 min(200, len) 条子集算 W-F1）
            eval_fn = (
                self._build_eval_fn(generator, dev)
                if cfg.training.eval_every_steps > 0 else None
            )
            train_summaries["sft"] = self._train_sft(
                model, tokenizer, adapter_manager, train, eval_fn=eval_fn
            )
        elif cfg.training.sft_enabled:
            logger.warning("[run] lora.enabled=false：跳过 SFT（无微调 prompting 基线）")

        # ---- 4) 辩论推理（dev）→ 偏好对 → DPO ----
        hetero_gen = None
        if cfg.debate.enabled and cfg.debate.agent2 == "hetero":
            het_model, het_tok = load_backbone(cfg.debate.hetero_model)
            hetero_gen = LLMGenerator(het_model, het_tok)
        runtime = build_runtime(cfg, generator, hetero_generator=hetero_gen)
        stages = PipelineAssembler.assemble(cfg, runtime)

        if cfg.training.dpo_enabled:
            # S6：dev 辩论生成偏好对前切 eval（偏好对不受 dropout 污染）
            model.eval()
            dev_states = [run_pipeline(stages, s) for s in dev]
            pairs = build_pairs(
                outcomes=[self._as_outcome(st) for st in dev_states],
                golds=[s.gold_label for s in dev],
                prompts=[st.prompt for st in dev_states],
                schema=self.schema,
            )
            logger.info("[run] DPO 偏好对 %d 组（dev 辩论产出）", len(pairs))
            if adapter_manager is not None:
                adapter_manager.activate("main")
            dpo = DPOTrainer(model, tokenizer, cfg.training, self.run_dir / "dpo")
            train_summaries["dpo"] = dpo.train(pairs)

        # ---- 4.5) 仅训练模式：adapter 已在 _train_sft / DPO 内落盘，跳过 test 推理 ----
        if cfg.skip_eval:
            results = {
                "run_id": self.run_id,
                "seed": self.seed,
                "config_name": cfg.name,
                "train_summaries": train_summaries,
                "synthesis_stats": synthesis_stats,
                "peak_memory_gib": round(_peak_memory_gib(), 2),
                "wall_time_seconds": round(time.time() - t0, 1),
                "skip_eval": True,
            }
            with open(self.run_dir / "results.json", "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=2)
            logger.info("[run] %s skip_eval：adapter 已保存，跳过 test 推理（%.1fs）",
                        self.run_id, results["wall_time_seconds"])
            return results

        # ---- 5) test 推理 + 双口径指标 ----
        model.eval()
        test_states = [run_pipeline(stages, s) for s in test]
        preds = [st.final_label for st in test_states]
        golds = [s.gold_label for s in test]
        results: dict = {
            "run_id": self.run_id,
            "seed": self.seed,
            "config_name": cfg.name,
            "smoke_samples": cfg.smoke_samples,  # S6：smoke 口径随结果落盘
            "metrics_full": compute_metrics(preds, golds, self.schema.labels),
            "metrics_real_only": self._real_only_metrics(test_states),
            "error_pairs": [
                {"gold": g, "pred": p, "count": c}
                for g, p, c in error_pair_counts(preds, golds)
            ],
        }

        # ---- 6) 辩论过程指标（M1 验收：收敛率/错误共识率/兜底触发率/兜底纠错率）----
        debate_outcomes = [st.debate_outcome for st in test_states if st.debate_outcome is not None]
        if debate_outcomes:
            sub_golds = [
                g for g, st in zip(golds, test_states) if st.debate_outcome is not None
            ]
            # 单 Agent 近似基线：辩论首轮 Agent1 标签（S6：first_round_label，
            # D 组新字段缺失/为 None 时回退提出方末轮输出）
            single_labels = [
                self._single_agent_label(st)
                for st in test_states if st.debate_outcome is not None
            ]
            results["debate"] = evaluate_debate(debate_outcomes, sub_golds, single_labels)
            # S6：辩论样本平均 LLM 生成调用次数（D 组 generation_calls，缺失按 0）
            gen_calls = [float(getattr(o, "generation_calls", 0) or 0) for o in debate_outcomes]
            results["debate"]["mean_generation_calls"] = sum(gen_calls) / len(gen_calls)

        # ---- 7) 路由统计 / Oracle 上界 ----
        if cfg.routing.enabled:
            modes = [st.mode.value if st.mode else "none" for st in test_states]
            correct = [p == g for p, g in zip(preds, golds)]
            results["routing"] = route_metrics(modes, correct)
            if cfg.routing.oracle_mode:
                results["oracle"] = self._oracle(stages, test, golds, preds)

        # ---- 8) 落盘 ----
        results["train_summaries"] = train_summaries
        results["synthesis_stats"] = synthesis_stats
        results["peak_memory_gib"] = round(_peak_memory_gib(), 2)
        results["wall_time_seconds"] = round(time.time() - t0, 1)
        self._write_predictions(test_states)
        with open(self.run_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[run] %s 完成：W-F1=%.4f Macro-F1=%.4f（%.1fs，峰值显存 %.2f GiB）",
            self.run_id,
            results["metrics_full"]["wf1"],
            results["metrics_full"]["macro_f1"],
            results["wall_time_seconds"],
            results["peak_memory_gib"],
        )
        return results

    # ------------------------------------------------------------------
    # 子步骤
    # ------------------------------------------------------------------

    def _train_sft(self, model, tokenizer, adapter_manager: AdapterManager,
                   train: list[DialogueSample], eval_fn=None) -> dict:
        cfg = self.config
        ckpt_dir = self.run_dir / "checkpoints"
        class_weights = (
            class_balanced_weights([s.gold_label for s in train])
            if cfg.fewshot.class_balanced else None
        )
        focal_gamma = cfg.fewshot.focal_gamma if cfg.fewshot.focal_loss_enabled else None
        main_trainer = SFTTrainer(
            model, tokenizer, self.schema, cfg.training, ckpt_dir,
            adapter_manager=adapter_manager,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
        enabled_aux = [a for a in cfg.aux_tasks if a.enabled]
        if not enabled_aux:
            summary = main_trainer.train(train, eval_fn=eval_fn)
        else:
            from ..aux_task import IntensityEstimator, MultitaskScheduler, SpeakerEstimator

            aux_trainers: dict = {}
            aux_samples: dict = {}
            aux_reports: dict = {}
            for aux in enabled_aux:
                if aux.name == "intensity":
                    head = IntensityEstimator()
                elif aux.name == "speaker_task":
                    head = SpeakerEstimator(
                        speakers=[s.context.target_speaker for s in train]
                    )
                else:
                    raise ValueError(f"未知辅助任务: {aux.name}")
                samples = head.to_aux_samples(train)
                if hasattr(head, "label_distribution") and aux.name == "intensity":
                    aux_reports["intensity_distribution"] = head.label_distribution(samples)
                adapter_manager.add(
                    aux.name,
                    LoRAConfig(r=aux.lora_r, alpha=aux.lora_alpha,
                               dropout=cfg.lora.dropout,
                               target_modules=cfg.lora.target_modules),
                )
                aux_trainer = SFTTrainer(
                    model, tokenizer, head.schema, cfg.training, ckpt_dir / aux.name,
                    adapter_manager=adapter_manager,
                    prompt_builder=head.prompt_builder,
                )
                aux_trainers[aux.name] = (aux_trainer, aux.loss_weight)
                aux_samples[aux.name] = samples
            # S6：多辅助任务 switch_steps 不一致时告警并取第一个任务的值
            if len({a.switch_steps for a in enabled_aux}) > 1:
                logger.warning(
                    "[run] 多辅助任务 switch_steps 不一致 %s，取第一个任务 %s 的值 %d",
                    {a.name: a.switch_steps for a in enabled_aux},
                    enabled_aux[0].name, enabled_aux[0].switch_steps,
                )
            scheduler = MultitaskScheduler(
                main_trainer, aux_trainers, adapter_manager,
                switch_steps=enabled_aux[0].switch_steps,
            )
            summary = scheduler.train(train, aux_samples, eval_fn=eval_fn)
            summary["aux"] = aux_reports
        # 保存 main adapter（推理/续训入口）
        adapter_manager.save("main", self.run_dir / "adapter_main")
        return summary

    @staticmethod
    def _as_outcome(state) -> DebateOutcome:
        """把单 Agent 路径的 InferenceState 包装成 DebateOutcome（偏好对构造统一入口）。"""
        if state.debate_outcome is not None:
            return state.debate_outcome
        o = state.single_output
        return DebateOutcome(
            final_label=o.label,
            confidence=o.confidence,
            final_reasoning=o.raw_text,
        )

    @staticmethod
    def _single_agent_label(state) -> str:
        """S6：单 Agent 基线标签 = 辩论首轮 Agent1 标签（first_round_label）。

        D 组 DebateOutcome 新字段缺失/为 None 时回退提出方（Agent1）末轮输出，
        再退到 final_label（与旧行为一致）。
        """
        o = state.debate_outcome
        first = getattr(o, "first_round_label", None)
        if first is not None:
            return first
        if o.proponent_output is not None:
            return o.proponent_output.label
        return state.final_label

    def _build_eval_fn(self, generator, dev: list[DialogueSample]):
        """S6：训练期轻量评估回调（SFTTrainer 按 eval_every_steps 间隔调用）。

        direct agent 对 dev 前 min(200, len(dev)) 条子集算 W-F1；
        train/eval 模式切换由调用方（SFTTrainer）负责。dev 为空时返回 None。
        """
        subset = dev[: min(200, len(dev))]
        if not subset:
            return None
        agent = DirectClassifyAgent(
            generator, self.schema, build_generation_kwargs(self.config)
        )
        golds = [s.gold_label for s in subset]

        def eval_fn(model) -> float:
            preds = [agent.infer(s.context).label for s in subset]
            return compute_metrics(preds, golds, self.schema.labels)["wf1"]

        return eval_fn

    def _real_only_metrics(self, states) -> dict:
        """红线 #6：仅真实样本口径（过滤 synthetic=true 的合成样本）。"""
        real = [st for st in states if not st.sample.synthetic]
        preds = [st.final_label for st in real]
        golds = [st.sample.gold_label for st in real]
        return compute_metrics(preds, golds, self.schema.labels)

    def _oracle(self, stages, test, golds, actual_preds) -> dict:
        """Oracle 上界：三模式分别强制跑一遍，量化路由可挖空间（V3 7.2）。

        S6：gap 与上界同口径（upper_bound_wf1 − wf1(actual_preds)，由
        OracleUpperBound.gap 保证），实际预测的 W-F1 一并落盘便于核对。
        """
        predictions: dict = {}
        for mode in RouteMode:
            states = [run_pipeline(stages, s, forced_mode=mode) for s in test]
            predictions[mode.value] = [st.final_label for st in states]
        oracle = OracleUpperBound(predictions, golds)
        return {
            "upper_bound_wf1": oracle.upper_bound(),
            "actual_wf1": compute_metrics(
                actual_preds, golds, self.schema.labels
            )["wf1"],
            "gap": oracle.gap(actual_preds),
        }

    def _write_predictions(self, states) -> None:
        with open(self.run_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for st in states:
                o = st.debate_outcome
                fh.write(json.dumps({
                    "dialogue_id": st.sample.context.dialogue_id,
                    "gold": st.sample.gold_label,
                    "pred": st.final_label,
                    "synthetic": st.sample.synthetic,
                    "mode": st.mode.value if st.mode else None,
                    "converged": o.converged if o else None,
                    "rounds": o.rounds if o else None,
                    "reask_triggered": o.reask_triggered if o else None,
                }, ensure_ascii=False) + "\n")

    def _write_run_meta(self) -> None:
        """红线 #3：config 快照 + git hash + 环境版本，任意 run 可精确复现。"""
        try:
            import torch

            torch_ver = torch.__version__
            cuda_ver = torch.version.cuda
        except ImportError:
            torch_ver, cuda_ver = "unavailable", "unavailable"
        meta = {
            "run_id": self.run_id,
            "seed": self.seed,
            "config": dataclasses.asdict(self.config),
            "git_hash": _git_hash(),
            "torch": torch_ver,
            "cuda": cuda_ver,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.run_dir / "run_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
