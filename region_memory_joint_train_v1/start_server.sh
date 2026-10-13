#!/usr/bin/env bash
# Start one new independent pipeline. No source installation or bank rebuilding.
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
REPO="${JOINT_REPO:-/root/shared-nvme/BRIGHT-master}"
V3_ROOT="${JOINT_V3_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665}"
RESULT_BASE="${JOINT_RESULT_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_V1}"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
RESULT="$RESULT_BASE/$RUN_ID"
LOG="${RESULT_BASE}/${RUN_ID}.log"
test -x "$PYTHON"
test -d "$REPO"
test -s "$V3_ROOT/keys/paired_bank.pt"
mkdir -p "$RESULT_BASE"
test ! -e "$RESULT"
test ! -e "$LOG"
export PYTHONDONTWRITEBYTECODE=1
nohup "$PYTHON" -u "$TOOL_DIR/run.py" full \
  --repo "$REPO" --v3-root "$V3_ROOT" --output "$RESULT" \
  --seed "${JOINT_SEED:-3407}" --workers "${JOINT_WORKERS:-4}" \
  --device "${JOINT_DEVICE:-cuda:0}" --topk "${JOINT_TOPK:-64}" \
  --chunk-size "${JOINT_CHUNK_SIZE:-64}" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '后台PID=%s\n工具目录=%s\n结果目录=%s\n日志=%s\n' "$PID" "$TOOL_DIR" "$RESULT" "$LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$LOG"
printf '读取报告："%s" "%s/run.py" report --root "%s"\n' "$PYTHON" "$TOOL_DIR" "$RESULT"
printf '启动仅表示后台进程已创建；请查看日志确认预检、真实smoke和正式训练状态。\n'
