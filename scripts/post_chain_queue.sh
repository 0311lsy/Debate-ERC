#!/usr/bin/env bash
# 过夜链后续 GPU 队列：等 5-seed 主链（45-49）完成后自动串行执行。
# 顺序：8-seed 聚合（CPU）→ self-consistency（单模型）→ hint 消融（双模型）
#       → Mistral-7B 第二异构家族（训练 + 终审）。
# 显存串行占用，绝不与主链并发；每步失败不阻断后续（标记 FAILED）。
set -u
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
LOG=outputs/post_chain_queue.log
PROP_S42=outputs/runs/base_sft_s42/adapter_main/main
MISTRAL_DIR=/home/lsy20252770/InstructERC/LLM_bases/Mistral-7B-v0.1

echo "[queue] $(date '+%F %T') 队列守护启动，等待主链完成标记..." >> "$LOG"
while ! grep -q "OVERNIGHT 5-SEED CHAIN DONE" outputs/overnight_seeds45_49.log 2>/dev/null; do
  sleep 120
done
echo "[queue] $(date '+%F %T') 主链完成，开始后续队列" >> "$LOG"

# 1) 8-seed 聚合 + 显著性（纯 CPU）
$PY -u scripts/aggregate_selective.py >> "$LOG" 2>&1 \
  || echo "[queue] aggregate_selective FAILED" >> "$LOG"

# 2) Self-Consistency 对照（单模型，s42，SC@3/SC@5）
$PY -u scripts/self_consistency.py --seed 42 --k 5 >> "$LOG" 2>&1 \
  || echo "[queue] self_consistency FAILED" >> "$LOG"

# 3) hint 消融：critic 不注入三类自检指引（双模型，固定 τ=0.65/margin=0.05，s42）
$PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
    --prop_adapter "$PROP_S42" \
    --critic_adapter outputs/hetero_critic/qwen7b_sft_s42/adapter_main/main \
    --no-critic-hint --tag qwen7b_s42_nohint >> "$LOG" 2>&1 \
  || echo "[queue] no_hint_ablation FAILED" >> "$LOG"

# 4) Mistral-7B 第二异构家族（依赖权重下载完成）
if [ -f "$MISTRAL_DIR/config.json" ]; then
  $PY -u scripts/train_critic_sft.py --backbone "$MISTRAL_DIR" --seed 42 \
      --out outputs/hetero_critic/mistral7b_sft_s42 >> "$LOG" 2>&1 \
    || echo "[queue] mistral_train FAILED" >> "$LOG"
  $PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
      --prop_adapter "$PROP_S42" --critic_base "$MISTRAL_DIR" \
      --critic_adapter outputs/hetero_critic/mistral7b_sft_s42/adapter_main/main \
      --tag mistral7b_s42 >> "$LOG" 2>&1 \
    || echo "[queue] mistral_eval FAILED" >> "$LOG"
else
  echo "[queue] Mistral 权重缺失，跳过步骤 4" >> "$LOG"
fi

echo "[queue] $(date '+%F %T') 后续队列全部结束" >> "$LOG"
