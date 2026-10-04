#!/usr/bin/env bash
set -u
cd /home/lsy20252770/Agent_Reason/Debate-ERC
PY=/home/lsy20252770/.conda/envs/instructerc/bin/python
LOG=outputs/dailydialog/aggregate.log

$PY -u scripts/dailydialog_aggregate.py > "$LOG" 2>&1
echo "########## DailyDialog 聚合完成 $(date '+%F %T') ##########" >> "$LOG"
# 自动回填到文档
$PY -u scripts/auto_update_dailydialog.py >> "$LOG" 2>&1 || true
echo "########## 回填完成 $(date '+%F %T') ##########" >> "$LOG"
