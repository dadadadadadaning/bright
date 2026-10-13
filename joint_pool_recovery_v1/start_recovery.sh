#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BASE=/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1
test -x "$PYTHON"
mkdir -p "$BASE"
LOG="$BASE/launcher_$(date +%Y%m%d_%H%M%S)_${BASHPID}.log"
test ! -e "$LOG"
nohup "$PYTHON" -u "$TOOL_DIR/recovery.py" --stage recover \
  > "$LOG" 2>&1 < /dev/null &
printf '恢复启动器PID=%s\n工具目录=%s\n启动日志=%s\n' "$!" "$TOOL_DIR" "$LOG"
printf '查看：tail -n 60 -f "%s"\n' "$LOG"
printf '先CPU检查，再等待指定SAR两任务终止及GPU空闲；候选修复未通过就停止。只启动一次。\n'
