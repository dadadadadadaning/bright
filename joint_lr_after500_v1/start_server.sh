#!/usr/bin/env bash
set -euo pipefail
BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_MODE="${1:-full}"
case "$BRIGHT_MODE" in full|check) ;; *) echo '用法：bash start_server.sh full 或 check'; exit 2;; esac
BRIGHT_SOURCE="${BRIGHT_TRAIN_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D}"
BRIGHT_ORIGINAL="${BRIGHT_ORIGINAL_TOOL:-/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2}"
test -x "$BRIGHT_PY"
"$BRIGHT_PY" "$BRIGHT_TOOL/study.py" plan --train-root "$BRIGHT_SOURCE" --tool-dir "$BRIGHT_ORIGINAL"
BRIGHT_BASE=/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_LR_AFTER500_V1
mkdir -p "$BRIGHT_BASE" /root/shared-nvme/BRIGHT-master/New
BRIGHT_RUN="$BRIGHT_BASE/run_$(date -u +%Y%m%d_%H%M%S)_${BRIGHT_MODE}_${BASHPID}"
test ! -e "$BRIGHT_RUN"
BRIGHT_LOG="$BRIGHT_RUN.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
fi
echo '此工具不等待 GPU 空闲，不停止已有任务；先真实 smoke，再执行本轮对照。'
nohup "$BRIGHT_PY" -u "$BRIGHT_TOOL/study.py" "$BRIGHT_MODE" \
  --train-root "$BRIGHT_SOURCE" --tool-dir "$BRIGHT_ORIGINAL" --output "$BRIGHT_RUN" \
  > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '%s\n' "$BRIGHT_RUN" > /root/shared-nvme/BRIGHT-master/New/joint_lr_after500_latest.txt
printf '%s\n' "$BRIGHT_PID" > "$BRIGHT_RUN.pid"
printf '后台PID=%s\n模式=%s\n结果目录=%s\n日志=%s\n' "$BRIGHT_PID" "$BRIGHT_MODE" "$BRIGHT_RUN" "$BRIGHT_LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$BRIGHT_LOG"
printf '读取报告："%s" "%s/study.py" report --root "%s"\n' "$BRIGHT_PY" "$BRIGHT_TOOL" "$BRIGHT_RUN"
echo '启动仅表示创建了后台进程；是否通过真实检查与完整 Val450，请看报告。'
