#!/bin/bash
# 补 seed 队列（单卡串行）：
#   段1-2: IEMOCAP s43/s44 全链（m0 → critic → 自身标定终审 → 冻结迁移终审）
#   段3-4: MELD hint 消融 s43/s44（--no-critic-hint，固定 τ=0.65/margin=0.05，与 s42 nohint 同口径）
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
CFG=configs/experiments/base_sft_iemocap.yaml

for S in 43 44; do
  PROP=outputs/runs/base_sft_iemocap_s${S}/adapter_main/main
  CRIT=outputs/hetero_critic/qwen7b_iemocap_sft_s${S}/adapter_main/main

  echo "########## [IEMOCAP s$S] m0 SFT $(date '+%H:%M:%S') ##########"
  $PY -u -m debate_erc.pipeline.run --config $CFG --seed $S --skip-eval

  echo "########## [IEMOCAP s$S] critic SFT $(date '+%H:%M:%S') ##########"
  $PY -u scripts/train_critic_sft.py --backbone $QWEN --dataset iemocap --seed $S \
      --out outputs/hetero_critic/qwen7b_iemocap_sft_s$S

  echo "########## [IEMOCAP s$S] 终审：自身 dev 标定 $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --config $CFG \
      --tag iemocap_qwen7b_s$S \
      --critic_base $QWEN \
      --critic_adapter $CRIT \
      --prop_adapter $PROP

  echo "########## [IEMOCAP s$S] 终审：冻结 MELD τ=0.65 迁移 $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
      --config $CFG \
      --tag iemocap_qwen7b_s${S}_transfer065 \
      --critic_base $QWEN \
      --critic_adapter $CRIT \
      --prop_adapter $PROP
done

for S in 43 44; do
  echo "########## [MELD hint 消融 s$S] 终审 $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --fixed-tau 0.65 --fixed-margin 0.05 --no-critic-hint \
      --tag qwen7b_s${S}_nohint \
      --critic_base $QWEN \
      --critic_adapter outputs/hetero_critic/qwen7b_sft_s${S}/adapter_main/main \
      --prop_adapter outputs/runs/base_sft_s${S}/adapter_main/main
done

echo "########## 补 seed 队列全部完成 $(date '+%H:%M:%S') ##########"
