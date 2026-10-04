#!/usr/bin/env bash
# M2(a) DailyDialog 冻结迁移评估：MELD 训练模型零调参迁移
# 前置：SC 多 seed 队列完成后手动启动或追加
set -u
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
LOG=outputs/dailydialog_eval.log

echo "[queue] DailyDialog 评估启动 $(date '+%F %T')" > "$LOG"

# 1. Proponent 贪心（LLaMA2-7B s42 adapter）
echo "########## Proponent 贪心 $(date '+%T') ##########" >> "$LOG"
$PY -u scripts/eval_greedy.py \
  --config configs/data/dailydialog.yaml \
  --checkpoints m0_sft=outputs/runs/base_sft_s42/adapter_main/main \
  --out outputs/dailydialog/greedy_proponent.json \
  >> "$LOG" 2>&1 \
  && echo "[queue] proponent 完成 $(date '+%T')" >> "$LOG" \
  || echo "[queue] proponent FAILED $(date '+%T')" >> "$LOG"

# 2. Proponent logprob 打分（置信度 → 门控输入）
echo "########## Proponent logprob $(date '+%T') ##########" >> "$LOG"
$PY -u scripts/eval_logprob.py \
  --config configs/data/dailydialog.yaml \
  --checkpoints m0_sft=outputs/runs/base_sft_s42/adapter_main/main \
  --out outputs/dailydialog/logprob_proponent.json \
  >> "$LOG" 2>&1 \
  && echo "[queue] proponent logprob 完成 $(date '+%T')" >> "$LOG" \
  || echo "[queue] proponent logprob FAILED $(date '+%T')" >> "$LOG"

# 3. Critic logprob 打分（Qwen2.5-7B）
echo "########## Critic logprob $(date '+%T') ##########" >> "$LOG"
$PY -u scripts/eval_logprob.py \
  --config configs/data/dailydialog_critic.yaml \
  --checkpoints critic=outputs/hetero_critic/qwen7b_sft_s42/adapter_main/main \
  --out outputs/dailydialog/logprob_critic.json \
  >> "$LOG" 2>&1 \
  && echo "[queue] critic logprob 完成 $(date '+%T')" >> "$LOG" \
  || echo "[queue] critic logprob FAILED $(date '+%T')" >> "$LOG"

echo "[queue] DailyDialog 评估全部完成 $(date '+%F %T')" >> "$LOG"

# ── DailyDialog 结果聚合与文档回填（后处理）──
echo "########## DailyDialog 聚合 $(date '+%F %T') ##########" >> "$LOG"
$PY -u scripts/dailydialog_aggregate.py > outputs/dailydialog/aggregate.log 2>&1 \
  && echo "[queue] DailyDialog 聚合完成 $(date '+%T')" >> "$LOG" \
  || echo "[queue] DailyDialog 聚合 FAILED $(date '+%T')" >> "$LOG"
