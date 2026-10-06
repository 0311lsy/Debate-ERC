#!/bin/bash
# C5 集成基线数据补齐：等 Mistral 8 种子队列结束后，dump s45–49 critic 全量 test 打分
# 规则：串行、失败即停
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
LOG=outputs/mistral_8seeds_queue.log

echo "########## critic all-test dump 队列等待 Mistral 8 种子 $(date '+%F %T') ##########" | tee -a "$LOG"
while pgrep -f "mistral_8seeds_queue.sh" >/dev/null; do sleep 60; done
# 确认 Mistral 队列成功结束而非失败
if ! grep -q "Mistral 8 种子队列全部完成" "$LOG"; then
  echo "!!! Mistral 8 种子队列未正常完成，dump 队列中止 $(date '+%F %T')" | tee -a "$LOG"
  exit 1
fi

echo "########## critic all-test dump 队列启动 $(date '+%F %T') ##########" | tee -a "$LOG"
for S in 45 46 47 48 49; do
  echo "########## [dump] critic s$S 全量 test $(date '+%F %T') ##########" | tee -a "$LOG"
  $PY -u scripts/dump_critic_all.py --backbone "$QWEN" \
      --adapter outputs/hetero_critic/qwen7b_sft_s${S}/adapter_main/main \
      --split test \
      --out outputs/selective_hetero/cross/critic${S}_all_test.jsonl >> "$LOG" 2>&1 \
    || { echo "[queue] dump_s${S} FAILED" >> "$LOG"; exit 1; }
done

echo "########## critic all-test dump 队列全部完成 $(date '+%F %T') ##########" | tee -a "$LOG"
