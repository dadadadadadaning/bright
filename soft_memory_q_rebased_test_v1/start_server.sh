#!/usr/bin/env bash
# Start only a completed, frozen Q-rebased A/B/C/D run on labelled Test878.
set -euo pipefail
BRIGHT_TEST_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIGHT_PY="${BRIGHT_PY:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
BRIGHT_TRAIN_ROOT="${BRIGHT_TRAIN_ROOT:?Set BRIGHT_TRAIN_ROOT to the completed full A/B/C/D run directory}"
BRIGHT_FIX2_TOOL="${BRIGHT_FIX2_TOOL:-/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2}"
BRIGHT_TEST_BASE="${BRIGHT_TEST_BASE:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_TEST_V1}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
test -x "$BRIGHT_PY"
test -s "$BRIGHT_TEST_TOOL/locked_test.py"

# Fast readiness check before creating any output. The Python runner then
# independently performs the full hash, checkpoint and Val450 evidence gates.
"$BRIGHT_PY" - "$BRIGHT_TRAIN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(relative):
    path = root / relative
    if not path.is_file():
        raise SystemExit('尚未完成四组正式训练，缺少：' + str(path))
    return json.loads(path.read_text())
state = read('workflow_state.json')
summary = read('train/summary.json')
final = read('train/final_val/result.json')
if (state.get('complete') is not True or state.get('full_validation_evidence') is not True
        or summary.get('complete') is not True or summary.get('full_validation_evidence') is not True
        or final.get('primary_endpoint') is not True or final.get('image_count') != 450):
    raise SystemExit('四组尚未完成 5000 步和正式末步 Val450；本次拒绝 Test。')
print('正式 Train/Val 快速状态检查通过；后台仍会逐文件验证后才读取 Test。')
PY

mkdir -p "$BRIGHT_TEST_BASE"
BRIGHT_TEST_LOG="$(mktemp "$BRIGHT_TEST_BASE/run_$(date +%Y%m%d_%H%M%S)_XXXXXXXX.log")"
BRIGHT_TEST_OUTPUT="${BRIGHT_TEST_LOG%.log}"
test ! -e "$BRIGHT_TEST_OUTPUT"
nohup "$BRIGHT_PY" -u "$BRIGHT_TEST_TOOL/locked_test.py" run \
  --train-root "$BRIGHT_TRAIN_ROOT" --tool "$BRIGHT_FIX2_TOOL" \
  --output "$BRIGHT_TEST_OUTPUT" > "$BRIGHT_TEST_LOG" 2>&1 < /dev/null &
BRIGHT_TEST_PID=$!
printf '后台PID=%s\n训练来源=%s\nTest结果目录=%s\n日志=%s\n' \
  "$BRIGHT_TEST_PID" "$BRIGHT_TRAIN_ROOT" "$BRIGHT_TEST_OUTPUT" "$BRIGHT_TEST_LOG"
printf '进度：tail -n 60 -f "%s"\n' "$BRIGHT_TEST_LOG"
printf '报告："%s" "%s/locked_test.py" report --output "%s"\n' \
  "$BRIGHT_PY" "$BRIGHT_TEST_TOOL" "$BRIGHT_TEST_OUTPUT"
printf 'PID 只代表后台进程已创建；Test878 正式结果以 complete/full_test_evidence 为准。\n'
