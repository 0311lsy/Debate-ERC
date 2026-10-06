#!/usr/bin/env python
"""真实推理延迟/显存基准（s42 MELD 生产适配器，单卡在线服务口径）。

测量每条话语的**生产路径**：贪心生成 8 token（取标签）+ 7 标签 teacher-forced
score_labels（门控置信度），与 eval_selective_hetero.py 完全一致。

报告：
  - primary(LLaMA2-7B) 与 critic(Qwen2.5-7B) 各自单流延迟 mean/P50/P95、吞吐；
  - 系统级：primary-only / 门控(τ=0.65，触发子串实测) / 全量审查(2.0×) 的
    逐样本延迟分布（双模型常驻显存，触发即串行第二次调用）；
  - 单模型 / 双模型常驻峰值显存；
  - 实测时间比 t_critic/t_primary，检验"1.27× 前向≈1.27× 墙钟"。

用法：python scripts/benchmark_latency.py [--n 400]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from debate_erc.agents.prompt_builder import build_classify_prompt  # noqa: E402
from debate_erc.config import load_config  # noqa: E402
from debate_erc.data import load_split  # noqa: E402
from debate_erc.deliberation import CRITIC_REASONING_HINT  # noqa: E402
from debate_erc.models import LLMGenerator, load_backbone  # noqa: E402
from debate_erc.utils.label_schema import get_schema  # noqa: E402
from debate_erc.utils.logging import get_logger  # noqa: E402
from debate_erc.utils.post_process import parse_label  # noqa: E402

logger = get_logger("debate_erc.scripts.benchmark_latency")
OUT = PROJECT_ROOT / "outputs/review_followups/latency_benchmark.json"
GEN_KW = {"do_sample": False, "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0}
TAU = 0.65


def pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def timed_call(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000  # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/experiments/base_sft.yaml"))
    ap.add_argument("--prop-adapter", default=str(PROJECT_ROOT / "outputs/runs/base_sft_s42/adapter_main/main"))
    ap.add_argument("--critic-base", default="/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B")
    ap.add_argument("--critic-adapter",
                    default=str(PROJECT_ROOT / "outputs/hetero_critic/qwen7b_sft_s42/adapter_main/main"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)
    labels = list(schema.labels)
    test = load_split(cfg.data.dataset, cfg.data.raw_path, "test", cfg.data.history_window)
    samples = test[: args.warmup + args.n]

    logger.info("加载 primary：%s + %s", cfg.backbone, args.prop_adapter)
    pm, ptok = load_backbone(cfg.backbone)
    pm = PeftModel.from_pretrained(pm, args.prop_adapter, adapter_name="m0")
    pm.set_adapter("m0"); pm.eval()
    prop = LLMGenerator(pm, ptok); prop.candidate_labels = labels
    torch.cuda.reset_peak_memory_stats()

    def prop_call(s):
        pr = build_classify_prompt(s.context, schema)
        y = parse_label(prop.generate(pr, **GEN_KW), schema)
        sc = prop.score_labels(pr, labels)
        return y, sc

    # warmup primary
    for s in samples[: args.warmup]:
        prop_call(s)
    mem_one = torch.cuda.max_memory_allocated() / 2**30

    logger.info("加载 critic：%s + %s", args.critic_base, args.critic_adapter)
    cm, ctok = load_backbone(args.critic_base)
    cm = PeftModel.from_pretrained(cm, args.critic_adapter, adapter_name="c")
    cm.set_adapter("c"); cm.eval()
    critic = LLMGenerator(cm, ctok); critic.candidate_labels = labels
    for s in samples[: args.warmup]:
        pr = build_classify_prompt(s.context, schema, reasoning_hint=CRITIC_REASONING_HINT)
        critic.generate(pr, **GEN_KW); critic.score_labels(pr, labels)
    torch.cuda.reset_peak_memory_stats()

    ta_ms, tb_ms, triggers, n_trig = [], [], [], 0
    with torch.no_grad():
        for i, s in enumerate(samples[args.warmup :]):
            pr = build_classify_prompt(s.context, schema)
            (ya, sca), la = timed_call(prop_call, s)
            p_a = sca[ya]
            ta_ms.append(la)
            if p_a < TAU:
                n_trig += 1
                pc = build_classify_prompt(s.context, schema, reasoning_hint=CRITIC_REASONING_HINT)
                (yb, scb), lb = timed_call(
                    lambda: (critic.generate(pc, **GEN_KW), critic.score_labels(pc, labels)))
                tb_ms.append(lb)
                triggers.append(True)
            else:
                triggers.append(False)
    mem_two = torch.cuda.max_memory_allocated() / 2**30

    # 系统级墙钟（串行）：门控仅触发样本付 critic；全量审查每条都付
    gated_ms, full_ms = [], []
    j = 0
    for i, la in enumerate(ta_ms):
        if triggers[i]:
            lb = tb_ms[j]; j += 1
            gated_ms.append(la + lb); full_ms.append(la + lb)
        else:
            gated_ms.append(la)
            # 未触发样本补一次 critic 计时（用已知 critic 延迟分布的均值做系统级估计会
            # 偏乐观/悲观——实测全量审查应直接逐条测量；此处 full 仅对触发集精确，
            # 整体用 mean(tb) 补齐并在结果中明确标注口径）
            full_ms.append(la + statistics.mean(tb_ms))

    def dist(xs):
        return {"mean_ms": round(statistics.mean(xs), 1),
                "p50_ms": round(pct(xs, 0.5), 1),
                "p95_ms": round(pct(xs, 0.95), 1),
                "throughput_utt_s": round(1000 / statistics.mean(xs), 2)}

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "n_measured": len(ta_ms), "tau": TAU,
        "measured_trigger_rate": round(n_trig / len(ta_ms), 4),
        "primary_llama2_7b": dist(ta_ms),
        "critic_qwen25_7b_triggered_only": dist(tb_ms),
        "system_primary_only": dist(ta_ms),
        "system_gated": dist(gated_ms),
        "system_full_review_2x_mean_imputed": dist(full_ms),
        "wallclock_ratio_critic_over_primary": round(
            statistics.mean(tb_ms) / statistics.mean(ta_ms), 3),
        "gated_wallclock_over_primary_empirical": round(
            statistics.mean(gated_ms) / statistics.mean(ta_ms), 3),
        "forward_cost_ratio_note": "前向口径 1+trigger=1.265；墙钟口径见上，受双模型切换/批内序列长度影响",
        "peak_mem_one_model_gb": round(mem_one, 2),
        "peak_mem_both_resident_gb": round(mem_two, 2),
        "dtype_note": "bf16 权重 + LoRA，batch=1 在线单流，generate 8 贪心 token + 7 标签打分",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("已保存 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
