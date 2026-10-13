#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
if [[ -n "${PYTHON:-}" ]]; then
  MEM_PYTHON="$PYTHON"
elif [[ -x /root/shared-nvme/conda-envs/bright4090/bin/python ]]; then
  MEM_PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
else
  MEM_PYTHON=python
fi
MEM_STAGE=plan
if [[ $# -gt 0 && "$1" != -* ]]; then
  MEM_STAGE="$1"
  shift
fi
cd -- "$REPO_DIR"
export PYTHONDONTWRITEBYTECODE=1
exec "$MEM_PYTHON" -m bda_benchmark.experiments.damageformer_srac_memory_rebuild_v3.run_pipeline --stage "$MEM_STAGE" "$@"
