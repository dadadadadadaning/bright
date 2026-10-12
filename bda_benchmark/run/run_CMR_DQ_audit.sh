#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_DQ_TRAIN_ONLY}"
OUTPUT="${OUTPUT:-${RESULT_ROOT}/CMR_DQ_contract_seed${SEED}.json}"

preflight() {
  python -m py_compile model/cmr/dynamic_queue.py script/standard_ML/audit_CMR_DQ_contract.py
  python -m unittest tests.test_cmr_dynamic_queue -v
  echo "CMR-DQ train-only queue preflight passed."
}

audit() {
  test -s "${TRAIN_CACHE}" || { echo "Missing CMR-P0 Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_DQ_contract.py \
    --train_cache "${TRAIN_CACHE}" \
    --output "${OUTPUT}" \
    --capacity_per_event_class "${CAPACITY_PER_EVENT_CLASS:-200}" \
    --positive_k "${POSITIVE_K:-5}" \
    --negative_k "${NEGATIVE_K:-10}" \
    --max_per_event "${MAX_NEIGHBORS_PER_EVENT:-2}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2; exit 2 ;;
esac

