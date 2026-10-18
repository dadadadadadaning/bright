#!/usr/bin/env bash
set -euo pipefail
BRIGHT_PY=/root/shared-nvme/conda-envs/bright4090/bin/python
BRIGHT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_FRESH_V2
BRIGHT_NEW=/root/shared-nvme/BRIGHT-master/New
BRIGHT_TOOL="$(cd "$(dirname "$0")" && pwd)"
"$BRIGHT_PY" "$BRIGHT_TOOL/fresh_chain.py" plan
mkdir -p "$BRIGHT_ROOT"
if ! flock -n "$BRIGHT_ROOT/active.lock" -c true; then
  printf '已有另一轮联合实验占用独占锁；请先查看该轮进度，不再启动重复任务。\n' >&2
  exit 2
fi
BRIGHT_TAG="$(date -u +%Y%m%d_%H%M%S)_$$"
BRIGHT_RUN="$BRIGHT_ROOT/run_$BRIGHT_TAG"
BRIGHT_LOG="$BRIGHT_RUN.log"
printf '%s\n' "$BRIGHT_RUN" > "$BRIGHT_NEW/joint_anchor_fresh_latest.txt"
printf '%s\n' "$BRIGHT_LOG" > "$BRIGHT_NEW/joint_anchor_fresh_log_latest.txt"
nohup "$BRIGHT_PY" -u "$BRIGHT_TOOL/fresh_chain.py" run --root "$BRIGHT_RUN" > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '后台PID=%s\n结果目录=%s\n日志=%s\n' "$BRIGHT_PID" "$BRIGHT_RUN" "$BRIGHT_LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$BRIGHT_LOG"
printf '读取报告：%s "%s/fresh_chain.py" report --root "%s"\n' "$BRIGHT_PY" "$BRIGHT_TOOL" "$BRIGHT_RUN"
