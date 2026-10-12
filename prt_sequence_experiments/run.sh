#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
CONTINUE_TO_TRAINING="${CONTINUE_TO_TRAINING:-0}"
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

ARGS=(--wait-gpu-idle)
case "$CONTINUE_TO_TRAINING" in
    0) ARGS+=(--checks-only) ;;
    1) ARGS+=(--continue-to-training) ;;
    *) echo "CONTINUE_TO_TRAINING must be 0 or 1"; exit 1 ;;
esac

"$PYTHON" "$TOOLS_DIR/run_sequence.py" --plan-only "${ARGS[@]}"
LOG="/root/shared-nvme/bright_prt_sequence_$(date +%Y%m%d_%H%M%S)_${BASHPID}.log"
touch "$LOG"
nohup "$PYTHON" -u "$TOOLS_DIR/run_sequence.py" "${ARGS[@]}" > "$LOG" 2>&1 &
printf '后台PID=%s\n日志=%s\n' "$!" "$LOG"
tail -n 60 -F "$LOG"
