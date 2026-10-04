#!/usr/bin/env bash
# 修补队列：重跑 post_chain_queue 中失败的两步（self_consistency 基线路径 + no_hint critic_base 缺失）。
set -u
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
LOG=outputs/redo_queue.log
PROP_S42=outputs/runs/base_sft_s42/adapter_main/main
QWEN7B_DIR=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B

echo "[redo] $(date '+%F %T') 修补队列启动" >> "$LOG"

# 1) Self-Consistency（修正基线路径 outputs/selective_hetero/qwen7b/）
$PY -u scripts/self_consistency.py --seed 42 --k 5 >> "$LOG" 2>&1 \
  || echo "[redo] self_consistency FAILED" >> "$LOG"

# 2) hint 消融（补 --critic_base Qwen-7B 路径）
$PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
    --prop_adapter "$PROP_S42" --critic_base "$QWEN7B_DIR" \
    --critic_adapter outputs/hetero_critic/qwen7b_sft_s42/adapter_main/main \
    --no-critic-hint --tag qwen7b_s42_nohint >> "$LOG" 2>&1 \
  || echo "[redo] no_hint_ablation FAILED" >> "$LOG"

echo "[redo] $(date '+%F %T') 修补队列结束" >> "$LOG"
