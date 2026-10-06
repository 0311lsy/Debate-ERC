#!/usr/bin/env python
"""门控叠加 InstructERC proponent v2：原生 greedy decode + 生成序列置信度。

修复 v1 的两个方法论错误：
1. y_a 改由 InstructERC 原生 prompt 的 greedy decode 产生（保持 66.29 基线），
   不再用 teacher-forced 候选标签 argmax（该口径把基线压到 32.78，使"增益"失真）。
2. p_a 取生成标签 token 序列的平均对数概率（generation confidence），
   与本文框架的候选标签后验是不同语义 —— 因此：
   a) 冻结 (τ=0.65, m=0.05) 直接叠加：检验零调参可移植性；
   b) 双阈值变体 (τ_a on p_a, τ_b on p_b)：避免跨尺度比较 p_b > p_a + m，
      在 dev 上网格校准后应用于 test（仅 dev，不调 test）。

critic 侧复用现有记录：test = cross/critic42_all_test.jsonl，
dev = qwen7b/dev_records.jsonl 的 (y_b, p_b)。两侧均按 idx 对齐（已验证 0 mismatch）。

用法：
    python scripts/instructerc_gate_stacking_v2.py --phase generate   # GPU
    python scripts/instructerc_gate_stacking_v2.py --phase gate       # CPU
    python scripts/instructerc_gate_stacking_v2.py --phase all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "/home/lsy20252770/InstructERC/LLM_bases/LLaMA2"
ADAPTER = PROJECT_ROOT / "outputs/instructerc_adapter_ep8"
DATA_DIR = Path("/home/lsy20252770/Agent_Reason/InstructERC/data/processed/meld_window")
CRITIC_TEST = PROJECT_ROOT / "outputs/selective_hetero/cross/critic42_all_test.jsonl"
CRITIC_DEV = PROJECT_ROOT / "outputs/selective_hetero/qwen7b/dev_records.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs/instructerc_gated_v2"

IE_LABELS = ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"]
IE_TO_OURS = {"sad": "sadness", "angry": "anger"}
LABELS_OURS = ["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"]

TAU_FROZEN, MARGIN_FROZEN = 0.65, 0.05


# ── Phase 1: GPU 生成 ────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, LlamaForCausalLM
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    return model, tokenizer


def parse_ie_label(text: str) -> str:
    t = text.strip().lower().split("\n")[0].strip().strip(".")
    for l in IE_LABELS:  # 精确匹配优先
        if t == l:
            return IE_TO_OURS.get(l, l)
    for l in IE_LABELS:  # 前缀/包含兜底
        if l in t:
            return IE_TO_OURS.get(l, l)
    return "neutral"


def generate_split(model, tokenizer, split: str) -> list[dict]:
    data_file = DATA_DIR / ("test.json" if split == "test" else "valid.json")
    data = [json.loads(l) for l in open(data_file)]
    print(f"=== {split}: {len(data)} samples ===")
    records = []
    t0 = time.time()
    for i, d in enumerate(data):
        inputs = tokenizer(d["input"], return_tensors="pt",
                           truncation=True, max_length=1015).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id)
        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        logprobs = []
        for t, score in enumerate(out.scores):
            lp = F.log_softmax(score[0].float(), dim=-1)
            logprobs.append(lp[gen_ids[t]].item())
        # 置信度：排除末尾 EOS 后 label token 的平均对数概率
        lp_label = logprobs
        if len(logprobs) > 1 and gen_ids[-1].item() == tokenizer.eos_token_id:
            lp_label = logprobs[:-1]
        p_a = float(torch.tensor(lp_label).mean().exp()) if lp_label else 0.0
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        records.append({
            "idx": i,
            "gold": IE_TO_OURS.get(d["target"], d["target"]),
            "y_a": parse_ie_label(gen_text),
            "p_a": round(p_a, 6),
            "gen_text": gen_text.strip(),
        })
        if (i + 1) % 300 == 0 or i == len(data) - 1:
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i+1}/{len(data)} ({rate:.2f}s/条, ETA {rate*(len(data)-i-1):.0f}s)")
    return records


# ── Phase 2: CPU 门控 ────────────────────────────────────────────────

def wf1(golds, preds):
    from sklearn.metrics import f1_score
    return f1_score(golds, preds, average="weighted", labels=LABELS_OURS)


def merge_critic(prop_records: list[dict], critic_file: Path, key_b: bool):
    """把 critic 的 (y_b, p_b) 按 idx 并入 proponent 记录。
    key_b=True 表示 critic 文件含 y_b/p_b 字段；否则整行即 critic 记录。"""
    critic = {}
    with open(critic_file) as f:
        for line in f:
            r = json.loads(line)
            critic[r["idx"]] = r
    merged = []
    for r in prop_records:
        cr = critic.get(r["idx"])
        if cr is None:
            continue
        merged.append({**r, "y_b": cr["y_b"], "p_b": cr["p_b"]})
    return merged


def apply_margin_rule(merged, tau, margin):
    """冻结 margin 规则：flip iff p_a<tau and y_b!=y_a and p_b>p_a+margin"""
    preds, n_trig, n_flip, rescue, harm = [], 0, 0, 0, 0
    for r in merged:
        flip = (r["p_a"] < tau and r["y_b"] != r["y_a"]
                and r["p_b"] > r["p_a"] + margin)
        n_trig += int(r["p_a"] < tau)
        n_flip += int(flip)
        if flip:
            rescue += int(r["y_b"] == r["gold"] and r["y_a"] != r["gold"])
            harm += int(r["y_b"] != r["gold"] and r["y_a"] == r["gold"])
        preds.append(r["y_b"] if flip else r["y_a"])
    return preds, {"trigger_rate": n_trig / len(merged), "n_flip": n_flip,
                   "rescue": rescue, "harm": harm}


def apply_two_threshold(merged, tau_a, tau_b):
    """双阈值规则：flip iff p_a<tau_a and y_b!=y_a and p_b>tau_b"""
    preds, n_trig, n_flip, rescue, harm = [], 0, 0, 0, 0
    for r in merged:
        flip = (r["p_a"] < tau_a and r["y_b"] != r["y_a"] and r["p_b"] > tau_b)
        n_trig += int(r["p_a"] < tau_a)
        n_flip += int(flip)
        if flip:
            rescue += int(r["y_b"] == r["gold"] and r["y_a"] != r["gold"])
            harm += int(r["y_b"] != r["gold"] and r["y_a"] == r["gold"])
        preds.append(r["y_b"] if flip else r["y_a"])
    return preds, {"trigger_rate": n_trig / len(merged), "n_flip": n_flip,
                   "rescue": rescue, "harm": harm}


def gate_phase():
    test_prop = [json.loads(l) for l in open(OUT_DIR / "test_records.jsonl")]
    dev_prop = [json.loads(l) for l in open(OUT_DIR / "dev_records.jsonl")]

    test = merge_critic(test_prop, CRITIC_TEST, key_b=True)
    dev = merge_critic(dev_prop, CRITIC_DEV, key_b=True)
    golds_t = [r["gold"] for r in test]
    golds_d = [r["gold"] for r in dev]

    summary = {"proponent": "InstructERC ep8 greedy + generation-logprob confidence",
               "critic": "Qwen2.5-7B s42", "n_test": len(test), "n_dev": len(dev)}

    # ── 基线 ──
    base_test = wf1(golds_t, [r["y_a"] for r in test])
    base_dev = wf1(golds_d, [r["y_a"] for r in dev])
    summary["base_wf1_test"] = round(base_test, 4)
    summary["base_wf1_dev"] = round(base_dev, 4)
    print(f"Base W-F1: test {base_test:.4f} | dev {base_dev:.4f}")

    # p_a 分布（与本文框架 SFT proponent 对比用）
    import statistics
    pas = [r["p_a"] for r in test]
    summary["p_a_dist"] = {
        "mean": round(statistics.mean(pas), 4),
        "median": round(statistics.median(pas), 4),
        "frac_below_0.65": round(sum(p < 0.65 for p in pas) / len(pas), 4),
        "frac_below_0.90": round(sum(p < 0.90 for p in pas) / len(pas), 4),
    }
    print(f"p_a dist: {summary['p_a_dist']}")

    # ── R0: oracle 上界（test，参照用）──
    oracle = [r["y_a"] if r["y_a"] == r["gold"] else r["y_b"] for r in test]
    summary["oracle_wf1_test"] = round(wf1(golds_t, oracle), 4)

    # ── R1: 冻结 margin 规则（零调参可移植性检验）──
    preds, st = apply_margin_rule(test, TAU_FROZEN, MARGIN_FROZEN)
    summary["frozen_margin"] = {"tau": TAU_FROZEN, "margin": MARGIN_FROZEN,
                                "wf1": round(wf1(golds_t, preds), 4),
                                "delta": round(wf1(golds_t, preds) - base_test, 4),
                                **{k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in st.items()}}
    print(f"Frozen margin (τ=0.65,m=0.05): {summary['frozen_margin']}")

    # ── R2: 双阈值，dev 校准 → test ──
    grid = []
    for tau_a in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995]:
        for tau_b in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds_d, st_d = apply_two_threshold(dev, tau_a, tau_b)
            grid.append({"tau_a": tau_a, "tau_b": tau_b,
                         "dev_wf1": wf1(golds_d, preds_d),
                         "dev_trigger": st_d["trigger_rate"]})
    grid.sort(key=lambda g: (g["dev_wf1"], -g["dev_trigger"]), reverse=True)
    best = grid[0]
    preds_t, st_t = apply_two_threshold(test, best["tau_a"], best["tau_b"])
    wf1_t = wf1(golds_t, preds_t)
    summary["two_threshold_dev_cal"] = {
        "tau_a": best["tau_a"], "tau_b": best["tau_b"],
        "dev_wf1": round(best["dev_wf1"], 4), "dev_trigger": round(best["dev_trigger"], 4),
        "test_wf1": round(wf1_t, 4), "delta": round(wf1_t - base_test, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in st_t.items()},
        "expected_cost_x": round(1 + st_t["trigger_rate"], 4),
    }
    print(f"Two-threshold (dev-cal τ_a={best['tau_a']},τ_b={best['tau_b']}): "
          f"{summary['two_threshold_dev_cal']}")

    # ── R3: 冻结 margin 规则但 τ 在 dev 上重校准（单参数）──
    grid1 = []
    for tau in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 0.995]:
        preds_d, _ = apply_margin_rule(dev, tau, MARGIN_FROZEN)
        grid1.append({"tau": tau, "dev_wf1": wf1(golds_d, preds_d)})
    grid1.sort(key=lambda g: g["dev_wf1"], reverse=True)
    bt = grid1[0]["tau"]
    preds_t, st_t = apply_margin_rule(test, bt, MARGIN_FROZEN)
    wf1_t = wf1(golds_t, preds_t)
    summary["margin_rule_dev_tau"] = {
        "tau": bt, "margin": MARGIN_FROZEN,
        "dev_wf1": round(grid1[0]["dev_wf1"], 4),
        "test_wf1": round(wf1_t, 4), "delta": round(wf1_t - base_test, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in st_t.items()},
        "expected_cost_x": round(1 + st_t["trigger_rate"], 4),
    }
    print(f"Margin rule dev-τ={bt}: {summary['margin_rule_dev_tau']}")

    (OUT_DIR / "summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_DIR/'summary_v2.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "gate", "all"], default="all")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("generate", "all"):
        model, tokenizer = load_model()
        for split in ["test", "dev"]:
            recs = generate_split(model, tokenizer, split)
            out = OUT_DIR / f"{split}_records.jsonl"
            with open(out, "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g = [r["gold"] for r in recs]
            y = [r["y_a"] for r in recs]
            print(f"  → {split} base W-F1 = {wf1(g, y):.4f}  saved {out}")

    if args.phase in ("gate", "all"):
        gate_phase()


if __name__ == "__main__":
    main()
