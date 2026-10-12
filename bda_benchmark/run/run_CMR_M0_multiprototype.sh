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
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M0_MULTIPROTOTYPE}"
OUTPUT="${OUTPUT:-${RESULT_ROOT}/CMR_M0_frozen_damaged_prototypes_seed${SEED}.json}"

preflight() {
  python -m py_compile model/cmr/multiprototype.py script/standard_ML/audit_CMR_M0_multiprototype.py
  python -m unittest tests.test_cmr_m0_multiprototype -v
  echo "CMR-M0 multi-prototype preflight passed."
}

audit() {
  test -s "${TRAIN_CACHE}" || { echo "Missing CMR-P0 Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_M0_multiprototype.py \
    --train_cache "${TRAIN_CACHE}" \
    --output "${OUTPUT}" \
    --clusters "${PROTOTYPE_COUNTS:-2,3,4,6}" \
    --seeds "${PROTOTYPE_SEEDS:-3407,3408,3409}" \
    --relation_pairs "${RELATION_PAIRS:-50000}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2; exit 2 ;;
esac

