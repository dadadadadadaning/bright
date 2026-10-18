#!/usr/bin/env bash
set -euo pipefail
BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_MODE="${1:-full}"
if [ "$#" -gt 0 ]; then shift; fi
case "$BRIGHT_MODE" in check|full) ;; *) echo '用法: bash start_server.sh check|full [run.py参数]'; exit 2;; esac
BRIGHT_BASE="${BRIGHT_RESULT_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_STUDY_V1}"
BRIGHT_NEW="${BRIGHT_NEW:-/root/shared-nvme/BRIGHT-master/New}"
test -x "$BRIGHT_PY"
"$BRIGHT_PY" "$BRIGHT_HERE/run.py" plan "$@"
mkdir -p "$BRIGHT_BASE" "$BRIGHT_NEW"
df -h "$BRIGHT_BASE"
BRIGHT_RUN="$BRIGHT_BASE/run_$(date -u +%Y%m%d_%H%M%S)_${BRIGHT_MODE}_${BASHPID}"
test ! -e "$BRIGHT_RUN"
BRIGHT_LOG="$BRIGHT_RUN.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
fi
printf '%s\n' '双槽、Memory、原NLL和Lovasz均保留；仅候选新增四类输出KL。' \
 '此入口不等待GPU，也不停止已有任务；full先做真实smoke，通过后从独立起点正式运行。'
nohup "$BRIGHT_PY" -u "$BRIGHT_HERE/run.py" "$BRIGHT_MODE" "$@" --output "$BRIGHT_RUN" > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '%s\n' "$BRIGHT_RUN" > "$BRIGHT_NEW/joint_anchor_latest.txt"
printf '%s\n' "$BRIGHT_PID" > "$BRIGHT_RUN.pid"
printf '后台PID=%s\n结果目录=%s\n日志=%s\n' "$BRIGHT_PID" "$BRIGHT_RUN" "$BRIGHT_LOG"
printf '看日志: tail -n 60 -f "%s"\n' "$BRIGHT_LOG"
printf '读报告: "%s" "%s/run.py" report --root "%s"\n' "$BRIGHT_PY" "$BRIGHT_HERE" "$BRIGHT_RUN"
printf '%s\n' '后台进程创建不等于真实前向成功；最终以末步Val450和终检为准。'
