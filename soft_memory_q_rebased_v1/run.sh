#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/shared-nvme/conda-envs/bright4090/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
exec "$PYTHON" -u "$TOOL_DIR/run.py" "$@"
