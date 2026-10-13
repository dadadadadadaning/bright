#!/usr/bin/env bash
# Creates only a NEW run. Default check builds small values and runs 20-step smoke.
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
MODE="${1:-check}"
if (( $# > 0 )); then shift; fi
case "$MODE" in check|full) ;; *) echo '首个参数必须为 check 或 full'; exit 2;; esac
REPO="${SOFT_JOINT_REPO:-/root/shared-nvme/BRIGHT-master}"
V3_ROOT="${SOFT_JOINT_V3_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665}"
RESULT_BASE="${SOFT_JOINT_RESULT_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_JOINT_V1}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
test -x "$PYTHON"
# Output belongs to this launcher, so an override cannot mislabel the log.
for ARG in "$@"; do
  case "$ARG" in --output|--output=*) echo '请用 SOFT_JOINT_RESULT_BASE 指定结果父目录；不要传 --output。'; exit 2;; esac
done
COMMON=(--repo "$REPO" --v3-root "$V3_ROOT" --seed "${SOFT_JOINT_SEED:-3407}"
        --workers "${SOFT_JOINT_WORKERS:-4}" --device "${SOFT_JOINT_DEVICE:-cuda:0}"
        --key-name "${SOFT_JOINT_KEY_NAME:-t128}" --topk 64
        --chunk-size "${SOFT_JOINT_CHUNK_SIZE:-32}")
# Reject incomplete releases, unavailable inputs and missing formal prerequisites
# before creating a log or background process. Never starts prior experiments.
"$PYTHON" - "$TOOL_DIR" "$MODE" "${COMMON[@]}" "$@" <<'PY'
import json, sys
from pathlib import Path
tool, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(tool))
import run
args = run.parser().parse_args([mode, '--output', '/unused/read-only-launch-check', *sys.argv[3:]])
record = run.validate_start(args)
print('启动前文件检查通过；尚未进行真实 CUDA 前向或训练。')
if mode == 'full':
    print('已验证旧三组结果：', record['baseline']['root'])
else:
    print('本次仅 check：预检、小库与 20 步 smoke；不会启动 5000 步正式训练。')
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
  printf '请留意已有 GPU 任务；此启动器不会等待空闲，也不会停止其他任务。\n'
fi
mkdir -p "$RESULT_BASE"
LOG="$(mktemp "$RESULT_BASE/run_$(date +%Y%m%d_%H%M%S)_${MODE}_XXXXXXXX.log")"
RESULT="${LOG%.log}"
test ! -e "$RESULT"
nohup "$PYTHON" -u "$TOOL_DIR/run.py" "$MODE" "${COMMON[@]}" "$@" \
  --output "$RESULT" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '后台PID=%s\n模式=%s\n工具目录=%s\n结果目录=%s\n日志=%s\n' "$PID" "$MODE" "$TOOL_DIR" "$RESULT" "$LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$LOG"
printf '读取报告："%s" "%s/run.py" report --root "%s"\n' "$PYTHON" "$TOOL_DIR" "$RESULT"
printf '启动仅表示后台进程已创建；检查通过不等于正式实验完成。CUDA 同步定位不修复原 pooling 故障。\n'
