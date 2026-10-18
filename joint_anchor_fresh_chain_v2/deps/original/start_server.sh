#!/usr/bin/env bash
# Start a new Q-rebased A/B/C/D run. A formal run includes true CUDA smoke first.
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
MODE="${1:-check}"
if (( $# > 0 )); then shift; fi
case "$MODE" in check|full) ;; *) echo '首个参数必须为 check 或 full'; exit 2;; esac

BRIGHT_REPO="${BRIGHT_REPO:-/root/shared-nvme/BRIGHT-master}"
BRIGHT_Q_CKPT="${BRIGHT_Q_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/run_20261014_003226_vdy7ey5t/sar_q/formal/train_baseline/last.pth}"
BRIGHT_A1_CKPT="${BRIGHT_A1_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth}"
BRIGHT_E0_METADATA="${BRIGHT_E0_METADATA:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json}"
BRIGHT_MEMBERSHIP="${BRIGHT_MEMBERSHIP:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665/keys}"
BRIGHT_RESULT_BASE="${BRIGHT_RESULT_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1}"
BRIGHT_DEVICE="${BRIGHT_DEVICE:-cuda:0}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

test -x "$BRIGHT_PY"
for BRIGHT_ARG in "$@"; do
  case "$BRIGHT_ARG" in --output|--output=*|--protocol|--protocol=*|--bank-root|--bank-root=*)
    echo '此启动器自行设置结果、来源协议和新库路径；请使用 BRIGHT_* 环境变量。'; exit 2;;
  esac
done
mkdir -p "$BRIGHT_RESULT_BASE"
BRIGHT_LOG="$(mktemp "$BRIGHT_RESULT_BASE/run_$(date +%Y%m%d_%H%M%S)_${MODE}_XXXXXXXX.log")"
BRIGHT_RESULT="${BRIGHT_LOG%.log}"
BRIGHT_PROTOCOL="${BRIGHT_RESULT}.source.json"
test ! -e "$BRIGHT_RESULT"

"$BRIGHT_PY" - "$TOOL_DIR" "$BRIGHT_Q_CKPT" "$BRIGHT_A1_CKPT" \
  "$BRIGHT_E0_METADATA" "$BRIGHT_PROTOCOL" "$BRIGHT_REPO" "$BRIGHT_DEVICE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run
run.verify_release()
from fresh_source import create_source_protocol
record = create_source_protocol(*sys.argv[2:])
if Path(record).resolve() != Path(sys.argv[5]).resolve():
    raise RuntimeError('Q-source protocol path differs from launch record')
print('新 Q 来源协议：', record)
PY

BRIGHT_COMMON=(--repo "$BRIGHT_REPO" --membership-root "$BRIGHT_MEMBERSHIP"
               --protocol "$BRIGHT_PROTOCOL" --seed "${BRIGHT_SEED:-3407}"
               --workers "${BRIGHT_WORKERS:-4}" --device "$BRIGHT_DEVICE"
               --key-name "${BRIGHT_KEY_NAME:-t128}" --topk 64
               --chunk-size "${BRIGHT_CHUNK_SIZE:-32}")
"$BRIGHT_PY" - "$TOOL_DIR" "$MODE" "${BRIGHT_COMMON[@]}" "$@" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run
args = run.parser().parse_args([sys.argv[2], '--output', '/unused/read-only-launch-check', *sys.argv[3:]])
record = run.validate_start(args)
print('启动前文件检查通过；旧 E0 文件未重放。')
print('来源协议 SHA256：', record['inputs']['source_protocol_sha256'])
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
fi
nohup "$BRIGHT_PY" -u "$TOOL_DIR/run.py" "$MODE" "${BRIGHT_COMMON[@]}" "$@" \
  --output "$BRIGHT_RESULT" > "$BRIGHT_LOG" 2>&1 < /dev/null &
BRIGHT_PID=$!
printf '后台PID=%s\n模式=%s\n工具目录=%s\n结果目录=%s\n日志=%s\n' \
  "$BRIGHT_PID" "$MODE" "$TOOL_DIR" "$BRIGHT_RESULT" "$BRIGHT_LOG"
printf '查看进度：tail -n 60 -f "%s"\n' "$BRIGHT_LOG"
printf '读取报告："%s" "%s/run.py" report --root "%s"\n' \
  "$BRIGHT_PY" "$TOOL_DIR" "$BRIGHT_RESULT"
printf 'full 会先做真实 CUDA smoke；最终成绩须等 5000 步和 Val450 完成。\n'
