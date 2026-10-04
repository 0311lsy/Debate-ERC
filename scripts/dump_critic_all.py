#!/usr/bin/env python
"""单个 critic 对指定 split 全样本输出 (y_b, p_b)，供因子交叉离线配对使用。

与 eval_selective_hetero.py 的 critic 调用逐字同口径（CRITIC_REASONING_HINT、
max_new_tokens=8 贪心、score_labels 候选内 softmax），只是不再做门控截断。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import torch
from peft import PeftModel
from debate_erc.agents.prompt_builder import build_classify_prompt
from debate_erc.config import load_config
from debate_erc.data import load_split
from debate_erc.deliberation import CRITIC_REASONING_HINT
from debate_erc.models import LLMGenerator, load_backbone
from debate_erc.utils.label_schema import get_schema
from debate_erc.utils.logging import get_logger
from debate_erc.utils.post_process import parse_label

logger = get_logger("debate_erc.scripts.dump_critic_all")
GEN_KW = {"do_sample": False, "max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(PROJECT_ROOT /
                    "configs/experiments/ablations/dpo_direct_classify.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    schema = get_schema(cfg.data.dataset)
    labels = list(schema.labels)

    model, tok = load_backbone(args.backbone)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="critic_dump")
    model.set_adapter("critic_dump")
    model.eval()
    critic = LLMGenerator(model, tok)
    critic.candidate_labels = labels

    samples = load_split(cfg.data.dataset, cfg.data.raw_path, args.split,
                         cfg.data.history_window)
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, s in enumerate(samples):
            prompt = build_classify_prompt(s.context, schema,
                                           reasoning_hint=CRITIC_REASONING_HINT)
            y_b = parse_label(critic.generate(prompt, **GEN_KW), schema)
            p_b = critic.score_labels(prompt, labels)[y_b]
            rows.append({"idx": i, "gold": s.gold_label,
                         "y_b": y_b, "p_b": round(p_b, 6)})
            if (i + 1) % 200 == 0:
                rate = (time.time() - t0) / (i + 1)
                logger.info("%s %d/%d（%.2fs/条，剩余 %.0fs）", args.split, i + 1,
                            len(samples), rate, rate * (len(samples) - i - 1))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    logger.info("已写出 %s（n=%d）", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
