#!/bin/bash
# Mistral critic s43/s44 扩展队列——等 gpu_queue_after_cross.sh 完成后自动接续
# 规则同主队列：串行、等显存释放、失败即停
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
MISTRAL=/home/lsy20252770/InstructERC/LLM_bases/Mistral-7B-v0.1
LOG=outputs/gpu_queue_after_cross.log

# 等待主队列脚本退出（通过检测 eval_selective_hetero 进程消失 + 日志标记）
echo "########## Mistral 扩展队列等待中 $(date '+%F %T') ##########" | tee -a "$LOG"
while pgrep -f "gpu_queue_after_cross.sh" >/dev/null; do sleep 60; done
# 额外等 B5 评估进程（eval_selective_hetero）退出
while pgrep -f "eval_selective_hetero.*halfdata" >/dev/null; do sleep 30; done

echo "########## Mistral 扩展队列启动 $(date '+%F %T') ##########" | tee -a "$LOG"
# 显存释放确认
for i in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  GPU memory used = ${USED} MiB" | tee -a "$LOG"
  [ "$USED" -lt 2000 ] && break
  sleep 30
done
if [ "${USED:-99999}" -ge 2000 ]; then
  echo "!!! GPU 未释放（${USED} MiB），Mistral 队列中止" | tee -a "$LOG"
  exit 1
fi

for S in 43 44; do
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

echo "########## Mistral 扩展队列全部完成 $(date '+%F %T') ##########" | tee -a "$LOG"
