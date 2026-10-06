#!/bin/bash
# EmoryNLP 追加 5 种子（45–49），与 42–44 完全同配方同协议。
# 目的：将 EmoryNLP 从 3 种子（+0.32±0.61, p=0.457 不显著）扩到 8 种子以提升统计功效。
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
CFG=configs/experiments/base_sft_emorynlp.yaml

for SEED in 45 46 47 48 49; do
  PROP=outputs/runs/base_sft_emorynlp_s${SEED}/adapter_main/main
  CRIT=outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}/adapter_main/main

  echo "########## [${SEED}/5] EmoryNLP primary SFT $(date '+%F %H:%M:%S') ##########"
  $PY -u -m debate_erc.pipeline.run --config $CFG --seed $SEED --skip-eval

  echo "########## [${SEED}/5] EmoryNLP Qwen7B critic SFT $(date '+%H:%M:%S') ##########"
  $PY -u scripts/train_critic_sft.py --backbone $QWEN --dataset emorynlp --seed $SEED \
      --out outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}

  echo "########## [${SEED}/5] EmoryNLP 终审（dev 标定 → test 锁定） $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --config $CFG \
      --tag emorynlp_qwen7b_s${SEED} \
      --critic_base $QWEN \
      --critic_adapter $CRIT \
      --prop_adapter $PROP
done

echo "########## EmoryNLP 追加 5 种子队列全部完成 $(date '+%F %H:%M:%S') ##########"
