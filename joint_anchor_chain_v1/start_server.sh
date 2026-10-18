#!/usr/bin/env bash
set -euo pipefail
BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_BASE="${BRIGHT_CHAIN_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_CHAIN_V1}"
BRIGHT_NEW="${BRIGHT_NEW:-/root/shared-nvme/BRIGHT-master/New}"
test -x "$BRIGHT_PY"
mkdir -p "$BRIGHT_BASE" "$BRIGHT_NEW"
BRIGHT_LOG="$(mktemp "$BRIGHT_BASE/run_$(date -u +%Y%m%d_%H%M%S)_XXXXXXXX.log")"
BRIGHT_RUN="${BRIGHT_LOG%.log}"
test ! -e "$BRIGHT_RUN"
printf '%s\n' "$BRIGHT_RUN" > "$BRIGHT_NEW/joint_anchor_chain_latest.txt"
printf '%s\n' "$BRIGHT_LOG" > "$BRIGHT_NEW/joint_anchor_chain_log_latest.txt"
nohup "$BRIGHT_PY" -u "$BRIGHT_HERE/chain.py" run "$@" --output "$BRIGHT_RUN" > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '%s\n' "$BRIGHT_PID" > "$BRIGHT_RUN.pid"
printf '后台PID=%s\n结果目录=%s\n日志=%s\n' "$BRIGHT_PID" "$BRIGHT_RUN" "$BRIGHT_LOG"
printf '%s\n' '顺序：依赖检查 → 必要时建库 → 新共同500步 → 两支至1500步＋Val450 → 双槽/Memory诊断。' \
 '任一步失败即停止；此入口不停止其他GPU任务。后台创建不等于训练开始。'
printf '看日志：tail -n 80 -f "%s"\n' "$BRIGHT_LOG"
printf '读状态："%s" "%s/chain.py" report --root "%s"\n' "$BRIGHT_PY" "$BRIGHT_HERE" "$BRIGHT_RUN"
