#!/usr/bin/env bash
set -euo pipefail

# M-CB0 is a fixed-workload, diagnostic human-review audit.  It trains no
# network, changes no P3 prediction, selects no budget on Validation, and
# refuses Test-derived caches.

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_REPLAY_R1/CMR_POOLING}"
TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/MEMORY_COMPLEMENT_BUDGET/CB0}"
RESULT="${RESULT:-${RESULT_ROOT}/M_CB0_fixed_budget_val_seed${SEED}.json}"

preflight() {
  for path in \
    script/standard_ML/mcb0_budget_runtime.py \
    script/standard_ML/audit_MCB0_budget.py \
    tests/test_mcb0_budget.py; do
    test -s "${path}" || { echo "Missing M-CB0 source: ${PROJECT_ROOT}/${path}" >&2; exit 3; }
  done
  python -m py_compile \
    script/standard_ML/mcb0_budget_runtime.py \
    script/standard_ML/audit_MCB0_budget.py \
    tests/test_mcb0_budget.py
  python -m unittest discover -s tests -p 'test_mcb0_budget.py' -v
  echo "M-CB0 fixed-budget preflight passed."
}

audit() {
  for path in "${TRAIN_P0}" "${VAL_P0}"; do
    test -s "${path}" || { echo "Missing M-CB0 input: ${path}" >&2; exit 4; }
  done
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_MCB0_budget.py \
    --train_p0_cache "${TRAIN_P0}" \
    --validation_p0_cache "${VAL_P0}" \
    --output "${RESULT}" \
    --device cuda:0 \
    --budgets "${BUDGETS:-50,100,200,350}" \
    --primary_budget "${PRIMARY_BUDGET:-200}" \
    --p3_fraction "${P3_FRACTION:-0.80}" \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --chunk_size "${QUERY_CHUNK:-512}" \
    --minimum_combined_damaged_gain "${MINIMUM_COMBINED_DAMAGED_GAIN:-1}" \
    --minimum_unique_damaged "${MINIMUM_UNIQUE_DAMAGED:-10}" \
    --minimum_unique_damaged_events "${MINIMUM_UNIQUE_DAMAGED_EVENTS:-5}" \
    --maximum_unique_damaged_event_share "${MAXIMUM_UNIQUE_DAMAGED_EVENT_SHARE:-0.40}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2; exit 2 ;;
esac
