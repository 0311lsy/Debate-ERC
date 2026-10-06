#!/bin/bash
# Mistral critic s45–49 补齐 8 种子队列——GPU 空闲直接启动
# 规则同主队列：串行、失败即停
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
MISTRAL=/home/lsy20252770/InstructERC/LLM_bases/Mistral-7B-v0.1
LOG=outputs/mistral_8seeds_queue.log

echo "########## Mistral 8 种子队列启动 $(date '+%F %T') ##########" | tee -a "$LOG"

for S in 45 46 47 48 49; do
  echo "########## [Mistral] critic 训练 s$S $(date '+%F %T') ##########" | tee -a "$LOG"
  $PY -u scripts/train_critic_sft.py --backbone "$MISTRAL" --seed $S \
      --out outputs/hetero_critic/mistral7b_sft_s${S} >> "$LOG" 2>&1 \
    || { echo "[queue] mistral_s${S}_train FAILED" >> "$LOG"; exit 1; }

  echo "########## [Mistral] 评估 s$S（冻结 τ=0.65/m=0.05） $(date '+%F %T') ##########" | tee -a "$LOG"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --fixed-tau 0.65 --fixed-margin 0.05 \
      --prop_adapter outputs/runs/base_sft_s${S}/adapter_main/main \
      --critic_base "$MISTRAL" \
      --critic_adapter outputs/hetero_critic/mistral7b_sft_s${S}/adapter_main/main \
      --tag mistral7b_s${S} >> "$LOG" 2>&1 \
    || { echo "[queue] mistral_s${S}_eval FAILED" >> "$LOG"; exit 1; }
done

echo "########## Mistral 8 种子队列全部完成 $(date '+%F %T') ##########" | tee -a "$LOG"
