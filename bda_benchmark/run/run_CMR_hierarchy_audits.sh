#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-e0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_CACHE="${VAL_CACHE:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_HIERARCHY}"
E0_RESULT="${E0_RESULT:-${RESULT_ROOT}/CMR_E0_occurrence_severity_val_seed${SEED}.json}"
M0B_RESULT="${M0B_RESULT:-${RESULT_ROOT}/CMR_M0B_event_capped_seed${SEED}.json}"
M0C_RESULT="${M0C_RESULT:-${RESULT_ROOT}/CMR_M0C_loeo_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    model/cmr/multiprototype.py \
    script/standard_ML/cmr_hierarchy_runtime.py \
    script/standard_ML/audit_CMR_E0_hierarchy.py \
    script/standard_ML/audit_CMR_M0B_event_capped.py \
    script/standard_ML/audit_CMR_M0C_loeo.py
  python -m unittest \
    tests.test_cmr_e0_hierarchy \
    tests.test_cmr_m0_multiprototype \
    tests.test_cmr_m0b_event_capped -v
  echo "CMR hierarchy E0/M0B/M0C preflight passed."
}

e0() {
  test -s "${VAL_CACHE}" || { echo "Missing Validation P0 cache: ${VAL_CACHE}" >&2; exit 3; }
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_E0_hierarchy.py \
    --validation_cache "${VAL_CACHE}" --output "${E0_RESULT}"
}

m0b() {
  test -s "${TRAIN_CACHE}" || { echo "Missing Train P0 cache: ${TRAIN_CACHE}" >&2; exit 3; }
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_M0B_event_capped.py \
    --train_cache "${TRAIN_CACHE}" --output "${M0B_RESULT}" \
    --clusters "${PROTOTYPE_COUNTS:-2,3,4}" \
    --seeds "${PROTOTYPE_SEEDS:-3407,3408,3409}" \
    --cap_per_event "${CAP_PER_EVENT:-200}" \
    --relation_pairs "${RELATION_PAIRS:-50000}"
}

m0c() {
  test -s "${M0B_RESULT}" || { echo "Missing M0B result: ${M0B_RESULT}" >&2; exit 4; }
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True else 5)' "${M0B_RESULT}" || {
    echo "M0B did not pass; M0C is correctly blocked." >&2
    exit 5
  }
  python -u script/standard_ML/audit_CMR_M0C_loeo.py \
    --train_cache "${TRAIN_CACHE}" --m0b_result "${M0B_RESULT}" \
    --output "${M0C_RESULT}" --cap_per_event "${CAP_PER_EVENT:-200}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  e0) e0 ;;
  m0b) m0b ;;
  m0c) m0c ;;
  diagnostics) preflight; e0; m0b ;;
  all) preflight; e0; m0b; m0c ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|e0|m0b|m0c|diagnostics|all" >&2; exit 2 ;;
esac

