#!/usr/bin/env python
"""门控叠加 InstructERC proponent：teacher-forced 打分 p_a → 冻结门控。

用 InstructERC epoch8 adapter 对 MELD test 的 7 个候选标签做受限分类打分，
得到 y_a / p_a，再与 critic（Qwen2.5-7B）的 y_b / p_b 按冻结门控规则合并。

标签映射：InstructERC 的 "sad"→"sadness", "angry"→"anger" 对齐本文 schema。

用法：
    python scripts/instructerc_gate_stacking.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── 路径 ──
BASE_MODEL = "/home/lsy20252770/InstructERC/LLM_bases/LLaMA2"
ADAPTER = PROJECT_ROOT / "outputs/instructerc_adapter_ep8"
TEST_JSON = "/home/lsy20252770/Agent_Reason/InstructERC/data/processed/meld_window/test.json"
CRITIC_JSONL = PROJECT_ROOT / "outputs/selective_hetero/cross/critic42_all_test.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs/instructerc_gated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── InstructERC 标签集与本文 schema 映射 ──
IE_LABELS = ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"]
IE_TO_OURS = {"sad": "sadness", "angry": "anger"}  # 其余同名

# ── 冻结门控参数 ──
TAU = 0.65
MARGIN = 0.05

MAX_LEN = 1024  # 训练时 max_length=1024


def load_model():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA2 无 pad token
    tokenizer.padding_side = "left"

    print("Loading base model ...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda"
    )
    print("Loading LoRA adapter ...")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    print("Model ready.")
    return model, tokenizer


def score_one_prompt(model, tokenizer, prompt: str, labels: list[str]):
    """对单个 prompt，计算每个候选标签的 teacher-forced logprob。

    返回 (n_labels,) 的 logprob 数组。
    """
    # 编码 prompt（含 BOS，与训练一致）
    prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_LEN - 10,
                           return_tensors="pt")
    p_ids = prompt_enc["input_ids"][0].tolist()

    # 编码 labels（不加 special tokens，与训练一致）
    label_enc = [tokenizer(l, add_special_tokens=False)["input_ids"] for l in labels]

    # 构造 7 个序列并 pad
    sequences = []
    label_starts = []
    for l_ids in label_enc:
        seq = p_ids + l_ids
        sequences.append(seq)
        label_starts.append(len(p_ids))

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    padded = []
    attn = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append([pad_id] * pad_len + seq)
        attn.append([0] * pad_len + [1] * len(seq))

    input_ids = torch.tensor(padded, device="cuda")
    attention_mask = torch.tensor(attn, device="cuda")

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # 计算每个 label 的 logprob
    logprobs_all = F.log_softmax(logits, dim=-1)
    scores = []
    for li, l_ids in enumerate(label_enc):
        start = label_starts[li]
        pad_len = max_len - len(sequences[li])
        actual_start = pad_len + start
        label_lp = 0.0
        for t in range(len(l_ids)):
            pos = actual_start + t  # 在 padded 序列中的位置
            token_id = l_ids[t]
            # logits[pos-1] 预测 token[pos]
            label_lp += logprobs_all[li, pos - 1, token_id].item()
        scores.append(label_lp)

    return torch.tensor(scores)


def main():
    t0 = time.time()

    # 加载 InstructERC test 数据
    test_data = []
    with open(TEST_JSON) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Test samples: {len(test_data)}")
    prompts = [d["input"] for d in test_data]
    golds_ie = [d["target"] for d in test_data]

    # 加载模型
    model, tokenizer = load_model()

    # 逐条打分
    all_scores = []
    for i, prompt in enumerate(prompts):
        scores = score_one_prompt(model, tokenizer, prompt, IE_LABELS)
        all_scores.append(scores)
        if (i + 1) % 200 == 0 or i == len(prompts) - 1:
            print(f"  scored {i + 1}/{len(prompts)}")

    scores = torch.stack(all_scores)  # (2610, 7)

    # softmax → p_a；argmax → y_a
    probs = F.softmax(scores, dim=-1)
    p_a_vals, y_a_idx = probs.max(dim=-1)
    y_a_ie = [IE_LABELS[i] for i in y_a_idx.tolist()]
    p_a_list = p_a_vals.tolist()

    # 映射到本文 schema
    def map_label(l):
        return IE_TO_OURS.get(l, l)

    y_a_ours = [map_label(l) for l in y_a_ie]
    golds_ours = [map_label(l) for l in golds_ie]

    # 验证：InstructERC 论文报告的 66.29 应能复现
    from sklearn.metrics import f1_score, accuracy_score
    ie_wf1 = f1_score(golds_ours, y_a_ours, average="weighted",
                      labels=["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"])
    ie_acc = accuracy_score(golds_ours, y_a_ours)
    print(f"\n=== InstructERC (logprob-argmax, our scoring) ===")
    print(f"  W-F1: {ie_wf1:.4f}  Acc: {ie_acc:.4f}")
    print(f"  (reference: greedy decode best-epoch W-F1 = 66.29)")

    # 加载 critic 打分
    critic_rows = {}
    with open(CRITIC_JSONL) as f:
        for line in f:
            r = json.loads(line)
            critic_rows[r["idx"]] = r

    # 冻结门控
    gated_preds = []
    flips = {"rescue": 0, "harm": 0, "neutral": 0}
    low_conf_count = 0
    adopted_count = 0

    for idx in range(len(test_data)):
        gold = golds_ours[idx]
        ya = y_a_ours[idx]
        pa = p_a_list[idx]

        cr = critic_rows.get(idx)
        if cr is None:
            gated_preds.append(ya)
            continue
        yb = cr["y_b"]
        pb = cr["p_b"]

        if pa < TAU:
            low_conf_count += 1

        if pa >= TAU:
            gated_preds.append(ya)
        elif yb != ya and pb > pa + MARGIN:
            gated_preds.append(yb)
            adopted_count += 1
            if yb == gold and ya != gold:
                flips["rescue"] += 1
            elif yb != gold and ya == gold:
                flips["harm"] += 1
            else:
                flips["neutral"] += 1
        else:
            gated_preds.append(ya)

    label_list = ["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"]
    gated_wf1 = f1_score(golds_ours, gated_preds, average="weighted", labels=label_list)
    gated_mf1 = f1_score(golds_ours, gated_preds, average="macro", labels=label_list)
    gated_acc = accuracy_score(golds_ours, gated_preds)

    # proponent logprob 指标（对照）
    base_wf1 = f1_score(golds_ours, y_a_ours, average="weighted", labels=label_list)
    base_mf1 = f1_score(golds_ours, y_a_ours, average="macro", labels=label_list)
    base_acc = accuracy_score(golds_ours, y_a_ours)

    summary = {
        "proponent": "InstructERC epoch8 (meld-only, LoRA r=16)",
        "proponent_scoring": "teacher-forced logprob over 7 candidate labels",
        "critic": "Qwen2.5-7B s42 (hetero_critic)",
        "n": len(test_data),
        "tau": TAU,
        "margin": MARGIN,
        "base_wf1": round(base_wf1, 4),
        "base_macro_f1": round(base_mf1, 4),
        "base_acc": round(base_acc, 4),
        "gated_wf1": round(gated_wf1, 4),
        "gated_macro_f1": round(gated_mf1, 4),
        "gated_acc": round(gated_acc, 4),
        "delta_wf1": round(gated_wf1 - base_wf1, 4),
        "trigger_rate": round(low_conf_count / len(test_data), 4),
        "adoption_rate": round(adopted_count / len(test_data), 4),
        "expected_cost_x": round(1 + low_conf_count / len(test_data), 4),
        "flips": flips,
        "ie_reported_wf1": 66.29,
        "elapsed_s": round(time.time() - t0, 1),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    # 保存逐条结果
    with open(OUT_DIR / "records.jsonl", "w") as f:
        for idx in range(len(test_data)):
            cr = critic_rows.get(idx, {})
            f.write(json.dumps({
                "idx": idx,
                "gold": golds_ours[idx],
                "y_a": y_a_ours[idx], "p_a": round(p_a_list[idx], 6),
                "y_b": cr.get("y_b"), "p_b": cr.get("p_b"),
                "gated": gated_preds[idx],
            }, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{'='*60}")
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
