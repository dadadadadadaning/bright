#!/usr/bin/env bash
set -euo pipefail

# Severity-aware successor of M1BG.  It reuses the locked train-only graph but
# removes Destroyed from the InfoNCE negatives.  DeltaCE is diagnostic only.
MODE="${1:-train}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${MODE}" == "preflight" ]]; then
  export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
  python -m py_compile \
    script/standard_ML/cmr_metric_utils.py \
    script/standard_ML/train_CMR_M1BG_graph.py
  python -m unittest \
    tests.test_cmr_m1bg_global_graph \
    tests.test_cmr_m1bgs_severity -v
  echo "CMR-M1BG-S severity-aware preflight passed."
  exit 0
fi

export NEGATIVE_CLASSES="0"
export GATE_PROTOCOL="m1bgs"
if [[ "${MODE}" == "smoke" ]]; then
  export EPOCHS="${EPOCHS:-1}"
  export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5}"
else
  export EPOCHS="${EPOCHS:-6}"
fi
export EVAL_EVERY="${EVAL_EVERY:-1}"
export MAXIMUM_DESTROYED_DROP="${MAXIMUM_DESTROYED_DROP:-0.01}"
export MINIMUM_NONNEGATIVE_EVENTS="${MINIMUM_NONNEGATIVE_EVENTS:-10}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M1BGS_SEVERITY}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/CMR_M1BGS_top${GRAPH_TOPK:-20}_epochs${EPOCHS}_seed${SEED:-3407}}"

bash run/run_CMR_M1BG_global_graph.sh "${MODE}"
