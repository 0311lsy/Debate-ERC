#!/usr/bin/env bash
# SC T=0.3 对照排队脚本：等待 seed_fillup_queue 全部完成（GPU 空闲）后运行。
# 回应审稿质疑："SC 负增益可能是 T=0.7 采样噪声过大所致"。
# 协议与 T=0.7 主对照严格一致（同 adapter/同 test/同 k=5/同破平规则），仅温度不同。
set -u
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
LOG=outputs/sc_t03_queue.log

echo "[queue] $(date '+%F %T') 等待 seed_fillup_queue 结束..." > "$LOG"
# 轮询等待：seed_fillup_queue.sh 进程消失即视为 GPU 空闲
while pgrep -f "seed_fillup_queue.sh" > /dev/null 2>&1; do
  sleep 60
done
echo "[queue] $(date '+%F %T') 前序队列结束，启动 SC T=0.3" >> "$LOG"

$PY -u scripts/self_consistency.py --seed 42 --k 5 --temperature 0.3 >> "$LOG" 2>&1 \
  && echo "[queue] $(date '+%F %T') SC T=0.3 完成" >> "$LOG" \
  || echo "[queue] $(date '+%F %T') SC T=0.3 FAILED" >> "$LOG"
