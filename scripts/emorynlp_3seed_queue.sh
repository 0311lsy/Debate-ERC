#!/bin/bash
# EmoryNLP（EmotionLines, Friends）第三数据集泛化队列：3 种子（42/43/44）
# 与 MELD/IEMOCAP 完全同配方：LLaMA2-7B primary + Qwen2.5-7B critic，LoRA r64/a128，3 epoch。
# 域内协议：每种子各自在 EmoryNLP dev 标定 τ×m，test 锁定终审（标签空间与 MELD
# 不相交，不做冻结门控迁移）。
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
CFG=configs/experiments/base_sft_emorynlp.yaml

for SEED in 42 43 44; do
  PROP=outputs/runs/base_sft_emorynlp_s${SEED}/adapter_main/main
  CRIT=outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}/adapter_main/main

  echo "########## [${SEED}/3] EmoryNLP primary SFT $(date '+%F %H:%M:%S') ##########"
  $PY -u -m debate_erc.pipeline.run --config $CFG --seed $SEED --skip-eval

  echo "########## [${SEED}/3] EmoryNLP Qwen7B critic SFT $(date '+%H:%M:%S') ##########"
  $PY -u scripts/train_critic_sft.py --backbone $QWEN --dataset emorynlp --seed $SEED \
      --out outputs/hetero_critic/qwen7b_emorynlp_sft_s${SEED}

  echo "########## [${SEED}/3] EmoryNLP 终审（dev 标定 → test 锁定） $(date '+%H:%M:%S') ##########"
  $PY -u scripts/eval_selective_hetero.py --mode both \
      --config $CFG \
      --tag emorynlp_qwen7b_s${SEED} \
      --critic_base $QWEN \
      --critic_adapter $CRIT \
      --prop_adapter $PROP
done

echo "########## EmoryNLP 3 种子队列全部完成 $(date '+%F %H:%M:%S') ##########"
