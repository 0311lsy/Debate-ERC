#!/usr/bin/env bash
# InstructERC 官方协议复现队列（MELD, LLaMA2-7B, LoRA, seeds 42/43/44）
#
# 仅执行 P1（官方训练+评估）。P2（四元组转储适配器）需在 P1 完成后
# 依据 main_new.py 的预测产物格式编写；P3（门控叠加）为纯 CPU 离线组合。
#
# 用法：
#   bash external_baselines/instructerc_reproduction/reproduce.sh           # 立即跑（需 GPU 空闲）
#   bash external_baselines/instructerc_reproduction/reproduce.sh --wait    # 等待当前 GPU 任务结束后再跑
set -u

IE_DIR=/home/lsy20252770/Agent_Reason/InstructERC
CODE_DIR=${IE_DIR}/code
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
MODEL=/home/lsy20252770/InstructERC/LLM_bases/LLaMA2
DATA=${IE_DIR}/data/processed/meld_window
OUT_BASE=/home/lsy20252770/Agent_Reason/Debate-ERC/external_baselines/instructerc_reproduction/results
LOG=/home/lsy20252770/Agent_Reason/Debate-ERC/outputs/instructerc_repro_queue.log
SEEDS=(42 43 44)

mkdir -p "$(dirname "$LOG")" "$OUT_BASE"

# 可选：等待 GPU 上无 python 训练/推理进程后再启动（不抢占正在运行的实验）
if [[ "${1:-}" == "--wait" ]]; then
  echo "[repro] 等待 GPU 空闲 $(date '+%F %T')" | tee -a "$LOG"
  while pgrep -f "self_consistency.py|eval_greedy.py|eval_logprob.py|train_critic_sft.py|main_new.py" >/dev/null; do
    sleep 60
  done
fi

echo "[repro] 启动 InstructERC 3-seed 复现 $(date '+%F %T')" | tee -a "$LOG"

cd "$CODE_DIR" || exit 1

for S in "${SEEDS[@]}"; do
  OUT=${OUT_BASE}/s${S}
  echo "########## InstructERC seed=${S} $(date '+%T') ##########" >> "$LOG"
  # 严格对齐官方 run_meld_lora.sh：1 epoch / lr2e-4 / bs16 / ga16 / maxlen1024
  $PY -u main_new.py \
    --dataset meld \
    --model_name_or_path "$MODEL" \
    --data_dir "$DATA" \
    --output_dir "$OUT" \
    --max_length 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 8 \
    --num_train_epochs 1 \
    --lora True \
    --learning_rate 2e-4 \
    --seed "$S" \
    --do_eval True \
    --do_train True \
    --statistic_mode True \
    >> "$LOG" 2>&1 \
    && echo "[repro] s${S} 完成 $(date '+%T')" >> "$LOG" \
    || { echo "[repro] s${S} FAILED $(date '+%T')" >> "$LOG"; exit 1; }
done

echo "########## P1 全部完成 $(date '+%F %T') ##########" >> "$LOG"
echo "[repro] 下一步：编写 P2 适配器（InstructERC 预测 → 四元组格式），随后 P3 门控叠加（纯 CPU）" >> "$LOG"
