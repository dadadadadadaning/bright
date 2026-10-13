#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
CONTINUE_TO_TRAINING="${CONTINUE_TO_TRAINING:-0}"
MODE="${1:-run}"
if [[ $# -gt 1 || "$MODE" != run && "$MODE" != --plan-only && "$MODE" != --preflight ]]; then
    echo "Usage: run.sh [--plan-only|--preflight]"
    exit 2
fi
test -x "$PYTHON"

"$PYTHON" - "$TOOLS_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
folder = Path(sys.argv[1])
hashes = json.loads((folder / "SHA256SUMS.json").read_text())
expected = {"README.md", "run.sh", "run_sequence.py", "initial_sequence.py",
            "test_run_sequence.py", "test_initial_sequence.py"}
if set(hashes) != expected:
    raise RuntimeError("Unexpected release file list")
for name, expected_hash in hashes.items():
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Release checksum mismatch: " + name)
print("Sequence package checksums passed")
PY

if [[ "$MODE" == --preflight ]]; then
    "$PYTHON" -m unittest discover -s "$TOOLS_DIR" -p 'test_*.py'
    "$PYTHON" "$TOOLS_DIR/run_sequence.py" --help >/dev/null
    "$PYTHON" "$TOOLS_DIR/initial_sequence.py" --help >/dev/null
    echo "CPU preflight complete. No GPU, model, Train, Val or training entry was started."
    exit 0
fi

ARGS=(--wait-gpu-idle)
ARGS+=(--repo "${REPO:-/root/shared-nvme/BRIGHT-master}")
ARGS+=(--python "$PYTHON")
ARGS+=(--failed-run "${FAILED_RUN:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34}")
ARGS+=(--night-tools "${NIGHT_TOOLS:-$TOOLS_DIR/../region_night}")
ARGS+=(--output-parent "${OUTPUT_PARENT:-/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_SEQUENCE_EXPERIMENTS_V1}")
case "$CONTINUE_TO_TRAINING" in
    0) ARGS+=(--checks-only) ;;
    1) ARGS+=(--continue-to-training) ;;
    *) echo "CONTINUE_TO_TRAINING must be 0 or 1"; exit 1 ;;
esac

"$PYTHON" "$TOOLS_DIR/run_sequence.py" --plan-only "${ARGS[@]}"
if [[ "$MODE" == --plan-only ]]; then
    exit 0
fi
LOG="/root/shared-nvme/bright_prt_sequence_$(date +%Y%m%d_%H%M%S)_${BASHPID}.log"
touch "$LOG"
nohup "$PYTHON" -u "$TOOLS_DIR/run_sequence.py" "${ARGS[@]}" > "$LOG" 2>&1 &
printf '后台PID=%s\n日志=%s\n' "$!" "$LOG"
tail -n 60 -F "$LOG"
