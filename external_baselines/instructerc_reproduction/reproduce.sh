#!/usr/bin/env bash
# 方案 A 骨架（未经作者拍板前禁止执行训练部分）：
# InstructERC unified-label 复现 = 数据准备（可先做）+ 自实现训练入口（尚不存在）
#
# 背景见同目录 README.md：论文 69.15 所用入口 main_Unilabel.py 官方未发布；
# 2026-09-28 的 meld-only 10ep 复现（66.29）协议错配，不能用于论文。
#
# 用法：
#   bash reproduce.sh prepare-data   # 仅生成 unified mixed 数据（CPU，可先验证）
#   bash reproduce.sh train          # 训练（阻断：自实现入口未完成且方案未拍板）
set -u

IE_CODE=/home/lsy20252770/Agent_Reason/InstructERC/code
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
PKL_ROOT=/home/lsy20252770/InstructERC/original_data
OUT_DATA=/home/lsy20252770/Agent_Reason/Debate-ERC/external_baselines/instructerc_reproduction/data/unified_label/mixed
PATCHED=/home/lsy20252770/Agent_Reason/Debate-ERC/external_baselines/instructerc_reproduction/data_process_mixed_patched.py

prepare_data() {
  # 前置检查：三个原始 pkl 必须齐全
  for d in meld iemocap EmoryNLP; do
    [[ -f "$PKL_ROOT/$d/$d.pkl" ]] || { echo "缺少 $PKL_ROOT/$d/$d.pkl"; exit 1; }
  done
  # 生成补丁脚本：仅替换原作者两处硬编码路径，其余逻辑不动
  mkdir -p "$(dirname "$PATCHED")" "$OUT_DATA"
  sed -e "s#/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/text_data#$PKL_ROOT#g" \
      -e "s#/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/leishanglin/LLMs_for_ERC/data/unified_label#$(dirname "$OUT_DATA")#g" \
      "$IE_CODE/data_process_mixed.py" > "$PATCHED"
  echo "[prepare] 补丁脚本：$PATCHED"
  cd "$IE_CODE" || exit 1
  # mode=mixed, window=12, data_percent=1.0（README 配比表中 69.15 对应全量行）
  "$PY" "$PATCHED" --mode mixed --historical_window 12 --data_percent 1.0
  echo "[prepare] 完成，检查 $OUT_DATA/{train,valid,test}.json 的样本数与 id 前缀分布"
}

case "${1:-}" in
  prepare-data)
    prepare_data
    ;;
  train)
    echo "[blocked] 方案 A 未拍板，且自实现 unified 训练入口（main_Unilabel 等价物）尚未编写。"
    echo "[blocked] 请先阅读 README.md 并在方案 A/B/C/D 中做出选择。"
    exit 2
    ;;
  *)
    echo "用法: bash reproduce.sh prepare-data | train"
    exit 1
    ;;
esac
