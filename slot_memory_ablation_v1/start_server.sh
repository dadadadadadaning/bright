#!/usr/bin/env bash
set -euo pipefail
BRIGHT_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_PY="${BRIGHT_PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_OUT="${BRIGHT_ABLATION_OUTPUT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_V1}"
BRIGHT_MODE="${1:-full}"
if (($#)); then shift; fi
if [[ "$BRIGHT_MODE" != full && "$BRIGHT_MODE" != smoke ]]; then
    echo "Usage: bash start_server.sh [full|smoke] [--group core|main|all] [--seeds 3407,3408,3409]"
    exit 2
fi
test -x "$BRIGHT_PY"
mkdir -p -- "$BRIGHT_OUT"
# A second lock guards the launcher itself, closing the background-start race.
exec 8>"$BRIGHT_OUT/launcher.lock"
if ! flock -n 8; then
    echo "已有此消融启动器运行；本次未重复启动。"
    exit 3
fi
"$BRIGHT_PY" "$BRIGHT_TOOL/ablation_runner.py" plan --output-base "$BRIGHT_OUT" "$@"
BRIGHT_LOG="$(mktemp "$BRIGHT_OUT/launcher_XXXXXXXX.log")"
nohup "$BRIGHT_PY" -u "$BRIGHT_TOOL/ablation_runner.py" "$BRIGHT_MODE" --output-base "$BRIGHT_OUT" "$@" >"$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '%s\n' "$BRIGHT_LOG" > "$BRIGHT_OUT/latest_launcher_log.txt"
printf '后台PID=%s\n日志=%s\n' "$BRIGHT_PID" "$BRIGHT_LOG"
printf '查看日志：tail -n 80 -f "%s"\n' "$BRIGHT_LOG"
printf '读取报告："%s" "%s/ablation_runner.py" report --output-base "%s"\n' "$BRIGHT_PY" "$BRIGHT_TOOL" "$BRIGHT_OUT"
printf '%s\n' 'full：CPU检查→独立CUDA smoke→正式训练与Val450；启动成功不等于训练完成。'
