#!/bin/bash
# IEMOCAP 泛化链 seed42：与 MELD 完全同配方（LLaMA2-7B m0 + Qwen2.5-7B critic，LoRA r64/a128，3 epoch）
# 段3：IEMOCAP 自身 dev 标定（标准协议）；段4：冻结 MELD 的 τ=0.65/margin=0.05 直接迁移（test 零调参证据）
set -e
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
QWEN=/home/lsy20252770/InstructERC/LLM_bases/Qwen2.5-7B
CFG=configs/experiments/base_sft_iemocap.yaml
PROP=outputs/runs/base_sft_iemocap_s42/adapter_main/main
CRIT=outputs/hetero_critic/qwen7b_iemocap_sft_s42/adapter_main/main

echo "########## [1/4] IEMOCAP m0 SFT $(date '+%H:%M:%S') ##########"
$PY -u -m debate_erc.pipeline.run --config $CFG --seed 42 --skip-eval

echo "########## [2/4] IEMOCAP Qwen7B critic SFT $(date '+%H:%M:%S') ##########"
$PY -u scripts/train_critic_sft.py --backbone $QWEN --dataset iemocap --seed 42 \
    --out outputs/hetero_critic/qwen7b_iemocap_sft_s42

echo "########## [3/4] 终审：IEMOCAP dev 标定 $(date '+%H:%M:%S') ##########"
$PY -u scripts/eval_selective_hetero.py --mode both \
    --config $CFG \
    --tag iemocap_qwen7b_s42 \
    --critic_base $QWEN \
    --critic_adapter $CRIT \
    --prop_adapter $PROP

echo "########## [4/4] 终审：冻结 MELD τ=0.65 迁移（零调参） $(date '+%H:%M:%S') ##########"
$PY -u scripts/eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 \
    --config $CFG \
    --tag iemocap_qwen7b_s42_transfer065 \
    --critic_base $QWEN \
    --critic_adapter $CRIT \
    --prop_adapter $PROP

echo "########## IEMOCAP 链全部完成 $(date '+%H:%M:%S') ##########"
