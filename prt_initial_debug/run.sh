#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
REPO="${REPO:-/root/shared-nvme/BRIGHT-master}"
FAILED_RUN="${FAILED_RUN:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34}"
OUTPUT_PARENT="${OUTPUT_PARENT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_INIT_DIAGNOSTIC_V1}"

test -x "$PYTHON"
test -f "$TOOLS_DIR/diagnose.py"
test -f "$TOOLS_DIR/diag_core.py"
test -f "$FAILED_RUN/prt_fit/initial_val/result.json"

"$PYTHON" - "$TOOLS_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
folder = Path(sys.argv[1])
hashes = json.loads((folder / "SHA256SUMS.json").read_text())
expected = {"README.md", "diag_core.py", "diagnose.py", "run.sh", "summarize.py",
            "test_diag_core.py", "test_diagnose.py"}
if set(hashes) != expected:
    raise RuntimeError("Unexpected release file list")
for name, expected_hash in hashes.items():
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Release checksum mismatch: " + name)
print("Diagnostic package checksums passed")
PY

GPU_PIDS="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [ -n "$GPU_PIDS" ]; then
    printf 'GPU 0 已有计算进程，请等它们结束后再运行本诊断。PID：\n%s\n' "$GPU_PIDS"
    exit 2
fi

mkdir -p "$OUTPUT_PARENT"
OUTPUT="$OUTPUT_PARENT/run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
LOG="$OUTPUT.log"
printf '诊断目录：%s\n日志：%s\n' "$OUTPUT" "$LOG"
if "$PYTHON" -u "$TOOLS_DIR/diagnose.py" \
    --repo "$REPO" --failed-run "$FAILED_RUN" --output "$OUTPUT" \
    --max-images 2 --head-precision-probes 2>&1 | tee "$LOG"; then
    "$PYTHON" "$TOOLS_DIR/summarize.py" "$OUTPUT/result.json" | tee -a "$LOG"
else
    RESULT_CODE=$?
    if [ -f "$OUTPUT/result.json" ]; then
        "$PYTHON" "$TOOLS_DIR/summarize.py" "$OUTPUT/result.json" || true
    fi
    exit "$RESULT_CODE"
fi
