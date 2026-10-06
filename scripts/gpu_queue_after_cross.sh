#!/bin/bash
# GPU 串行队列（用户 2026-10-05 授权）：
#   阶段 0: 等待角色互换 s43–49（PID 由 WAIT_PID 指定，默认 52658）释放 GPU
#   阶段 1（A3）: hint 消融补 s45–s49 nohint 评估（mode both，冻结 τ=0.65/m=0.05，
#                与 qwen7b_s42/43/44_nohint 严格同口径；adapter 已存在，纯评估）
#   阶段 2（B5）: 数据独立性对照——Qwen2.5-7B critic 用 50% MELD train 子集
#                （seed=42 确定性划分前 4994 条），6 epoch 控制总更新步数
#                ≈939（与全量 3 epoch 可比），其余超参与主协议一致；
#                proponent 仍为现有全量 base_sft_s42。训练后冻结门控评估。
# 任何一步失败即停止（set -e），不静默续跑。
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
WAIT_PID=${1:-52658}
LOG=outputs/gpu_queue_after_cross.log

echo "########## 队列启动 $(date '+%F %T')，等待角色互换 PID $WAIT_PID ##########" | tee -a "$LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "########## 角色互换已结束 $(date '+%F %T')，等待 GPU 显存释放 ##########" | tee -a "$LOG"
for i in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  GPU memory used = ${USED} MiB" | tee -a "$LOG"
  [ "$USED" -lt 2000 ] && break
  sleep 30
done
if [ "$USED" -ge 2000 ]; then
  echo "!!! GPU 未释放（${USED} MiB），队列中止，请人工检查" | tee -a "$LOG"
  exit 1
fi

########## 阶段 1：A3 nohint s45–49 ##########
for S in 45 46 47 48 49; do
  echo "########## [A3] MELD nohint 消融 s$S 评估 $(date '+%F %T') ##########" | tee -a "$LOG"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --fixed-tau 0.65 --fixed-margin 0.05 --no-critic-hint \
      --tag qwen7b_s${S}_nohint \
      --critic_base "$QWEN" \
      --critic_adapter outputs/hetero_critic/qwen7b_sft_s${S}/adapter_main/main \
      --prop_adapter outputs/runs/base_sft_s${S}/adapter_main/main >> "$LOG" 2>&1
  echo "########## [A3] s$S 完成 $(date '+%F %T') ##########" | tee -a "$LOG"
done

########## 阶段 2：B5 半数据 critic ##########
echo "########## [B5] critic 50% 数据子集训练（6 epoch，seed=42） $(date '+%F %T') ##########" | tee -a "$LOG"
$PY -u scripts/train_critic_sft.py --backbone "$QWEN" --seed 42 \
    --train-fraction 0.5 --epochs 6 \
    --out outputs/hetero_critic/qwen7b_sft_s42_halfdata >> "$LOG" 2>&1

echo "########## [B5] 半数据 critic 终审（冻结 τ=0.65） $(date '+%F %T') ##########" | tee -a "$LOG"
$PY -u scripts/eval_selective_hetero.py --mode both \
    --fixed-tau 0.65 --fixed-margin 0.05 \
    --tag qwen7b_s42_halfdata \
    --critic_base "$QWEN" \
    --critic_adapter outputs/hetero_critic/qwen7b_sft_s42_halfdata/adapter_main/main \
    --prop_adapter outputs/runs/base_sft_s42/adapter_main/main >> "$LOG" 2>&1

echo "########## GPU 队列全部完成 $(date '+%F %T') ##########" | tee -a "$LOG"
