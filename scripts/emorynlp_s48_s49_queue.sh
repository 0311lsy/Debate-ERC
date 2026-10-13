#!/bin/bash
# EmoryNLP 补齐 s48/s49（10-06 队列中断，s48 primary SFT step~200 被杀，s49 未开始）。
# 与 s42-47 完全同配方同协议，完成后 8 种子口径齐全。
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
CFG=configs/experiments/base_sft_emorynlp.yaml

for SEED in 48 49; do
  PROP=outputs/runs/base_sft_emorynlp_s${SEED}/adapter_main/main
  CRIT=outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}/adapter_main/main

  echo "########## [${SEED}] EmoryNLP primary SFT $(date '+%F %H:%M:%S') ##########"
  $PY -u -m debate_erc.pipeline.run --config $CFG --seed $SEED --skip-eval

  echo "########## [${SEED}] EmoryNLP Qwen7B critic SFT $(date '+%H:%M:%S') ##########"
  $PY -u scripts/train_critic_sft.py --backbone $QWEN --dataset emorynlp --seed $SEED \
      --out outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}

  echo "########## [${SEED}] EmoryNLP 终审（dev 标定 → test 锁定） $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --config $CFG \
      --tag emorynlp_qwen7b_s${SEED} \
      --critic_base $QWEN \
      --critic_adapter $CRIT \
      --prop_adapter $PROP
done

echo "########## EmoryNLP s48-49 补齐队列全部完成 $(date '+%F %H:%M:%S') ##########"
