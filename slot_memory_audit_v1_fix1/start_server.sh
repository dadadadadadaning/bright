#!/usr/bin/env bash
set -euo pipefail

BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_AUDIT_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_MODE="${1:-full}"
case "$BRIGHT_MODE" in
  full|check) ;;
  *) echo '用法：bash start_server.sh full 或 check'; exit 2 ;;
esac
BRIGHT_SOURCE="${BRIGHT_TRAIN_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D}"
BRIGHT_ORIGINAL="${BRIGHT_ORIGINAL_TOOL:-/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2}"
BRIGHT_STEP="${BRIGHT_AUDIT_STEP:-500}"
case "$BRIGHT_STEP" in
  500|1500|5000) ;;
  *) echo 'BRIGHT_AUDIT_STEP 仅支持 500、1500、5000。'; exit 2 ;;
esac
test -x "$BRIGHT_PY"
test -s "$BRIGHT_AUDIT_TOOL/audit.py"

"$BRIGHT_PY" "$BRIGHT_AUDIT_TOOL/audit.py" plan \
  --train-root "$BRIGHT_SOURCE" --tool-dir "$BRIGHT_ORIGINAL" --step "$BRIGHT_STEP"

BRIGHT_BASE=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_AUDIT_V1
BRIGHT_MARKER=/root/shared-nvme/BRIGHT-master/New/slot_memory_audit_latest.txt
mkdir -p "$BRIGHT_BASE" /root/shared-nvme/BRIGHT-master/New
df -h "$BRIGHT_BASE"
BRIGHT_RUN="$BRIGHT_BASE/run_$(date +%Y%m%d_%H%M%S)_step${BRIGHT_STEP}_${BRIGHT_MODE}_${BASHPID}"
test ! -e "$BRIGHT_RUN"
BRIGHT_LOG="$BRIGHT_RUN.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
fi
echo '本轮仅诊断已有权重：不训练、不更新参数、不访问 Test、不重建案例库。'
echo '此工具不等待 GPU 空闲，也不停止已有任务。'
if [ "$BRIGHT_MODE" = full ]; then
  echo 'full：CPU测试 → 两个不同事件的真实Val小样本检查 → 同一检查点完整Val450诊断。'
else
  echo 'check：CPU测试和两个不同事件的真实Val小样本检查；不执行完整Val450。'
fi
nohup "$BRIGHT_PY" -u "$BRIGHT_AUDIT_TOOL/audit.py" "$BRIGHT_MODE" \
  --train-root "$BRIGHT_SOURCE" --tool-dir "$BRIGHT_ORIGINAL" --step "$BRIGHT_STEP" \
  --output "$BRIGHT_RUN" > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '%s\n' "$BRIGHT_RUN" > "$BRIGHT_MARKER"
printf '%s\n' "$BRIGHT_PID" > "$BRIGHT_RUN.pid"
printf '后台PID=%s\n模式=%s\n检查点步数=%s\n工具目录=%s\n结果目录=%s\n日志=%s\n' \
  "$BRIGHT_PID" "$BRIGHT_MODE" "$BRIGHT_STEP" "$BRIGHT_AUDIT_TOOL" "$BRIGHT_RUN" "$BRIGHT_LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$BRIGHT_LOG"
printf '读取报告："%s" "%s/audit.py" report --root "%s"\n' \
  "$BRIGHT_PY" "$BRIGHT_AUDIT_TOOL" "$BRIGHT_RUN"
echo '启动只表示后台进程已创建；完整诊断结果以Val450和最终校验报告为准。'
