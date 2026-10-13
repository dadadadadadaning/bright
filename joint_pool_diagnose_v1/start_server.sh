#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
FAILED_RUN="${POOL_FAILED_RUN:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_V1/run_20261013_143518_3825644}"
OLD_TOOL="${POOL_OLD_TOOL:-/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1}"
REPO="${POOL_REPO:-/root/shared-nvme/BRIGHT-master}"
BASE="${POOL_RESULT_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_DIAG_V1}"
test -x "$PYTHON"
test -s "$OLD_TOOL/run.py"
test -s "$FAILED_RUN/train/manifest.json"
test -s "$FAILED_RUN/train/pool_debug/failure.json"
mkdir -p "$BASE"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
OUT="$BASE/$RUN_ID"
LOG="$BASE/$RUN_ID.log"
test ! -e "$OUT"
test ! -e "$LOG"
export PYTHONDONTWRITEBYTECODE=1
nohup "$PYTHON" -u "$TOOL_DIR/diagnose.py" all \
  --failed-run "$FAILED_RUN" --tool "$OLD_TOOL" --repo "$REPO" --output "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
printf '诊断PID=%s\n诊断工具=%s\n结果=%s\n日志=%s\n' "$!" "$TOOL_DIR" "$OUT" "$LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$LOG"
printf '读取报告："%s" "%s/diagnose.py" report --root "%s"\n' "$PYTHON" "$TOOL_DIR" "$OUT"
printf '仅诊断：不会更新权重、重建库或启动5000步训练；旧现场保留。\n'
