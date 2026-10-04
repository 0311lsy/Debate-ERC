#!/bin/bash
# 5 个独立种子对：m0_sft(LLaMA2) + qwen7b critic，固定 τ=0.65/margin=0.05（pooled-dev 锁定）
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
for S in 45 46 47 48 49; do
  echo "########## SEED $S m0 训练 $(date '+%H:%M:%S') ##########"
  $PY -u -m debate_erc.pipeline.run --config configs/experiments/base_sft.yaml --seed $S --skip-eval
  echo "########## SEED $S critic 训练 $(date '+%H:%M:%S') ##########"
  $PY -u scripts/train_critic_sft.py --backbone $QWEN --seed $S \
      --out outputs/hetero_critic/qwen7b_sft_s$S
  echo "########## SEED $S 终审（固定τ=0.65） $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
      --tag qwen7b_s${S}_shared \
      --critic_base $QWEN \
      --critic_adapter outputs/hetero_critic/qwen7b_sft_s$S/adapter_main/main \
      --prop_adapter outputs/runs/base_sft_s$S/adapter_main/main
  echo "########## SEED $S 全部完成 $(date '+%H:%M:%S') ##########"
done
echo "########## OVERNIGHT 5-SEED CHAIN DONE $(date '+%H:%M:%S') ##########"
