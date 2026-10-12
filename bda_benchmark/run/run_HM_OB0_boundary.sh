#!/usr/bin/env bash
set -euo pipefail

# Isolated, diagnostic-only HM-OB0 entry point.  It trains no network, changes
# no P3 prediction, selects nothing on Validation, and refuses Test caches.

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_REPLAY_R1/CMR_POOLING}"
TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/HIERARCHICAL_MEMORY/OB0}"
OB0_RESULT="${OB0_RESULT:-${RESULT_ROOT}/HM_OB0_train_locked_boundary_val_seed${SEED}.json}"

preflight() {
  for path in \
    script/standard_ML/hm_ob0_boundary_runtime.py \
    script/standard_ML/audit_HM_OB0_boundary.py \
    tests/test_hm_ob0_boundary.py; do
    test -s "${path}" || {
      echo "Missing HM-OB0 source file: ${PROJECT_ROOT}/${path}" >&2
      echo "Synchronise all HM-OB0 files before running preflight." >&2
      exit 3
    }
  done
  python -m py_compile \
    script/standard_ML/hm_ob0_boundary_runtime.py \
    script/standard_ML/audit_HM_OB0_boundary.py \
    tests/test_hm_ob0_boundary.py
  python -m unittest discover \
    -s tests \
    -p 'test_hm_ob0_boundary.py' \
    -v
  echo "HM-OB0 boundary audit preflight passed."
}

audit() {
  for path in "${TRAIN_P0}" "${VAL_P0}"; do
    test -s "${path}" || { echo "Missing OB0 input: ${path}" >&2; exit 3; }
  done
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_HM_OB0_boundary.py \
    --train_p0_cache "${TRAIN_P0}" \
    --validation_p0_cache "${VAL_P0}" \
    --output "${OB0_RESULT}" \
    --seed "${SEED}" \
    --cap_per_event_class "${CAP_PER_EVENT_CLASS:-500}" \
    --coverages "${COVERAGES:-0.005,0.01,0.025,0.05,0.10}" \
    --logistic_l2 "${LOGISTIC_L2:-0.01}" \
    --logistic_max_iter "${LOGISTIC_MAX_ITER:-200}" \
    --minimum_validation_candidates "${MINIMUM_VALIDATION_CANDIDATES:-100}" \
    --minimum_global_auc "${MINIMUM_GLOBAL_AUC:-0.70}" \
    --minimum_event_auc "${MINIMUM_EVENT_AUC:-0.60}" \
    --minimum_auc_gain_vs_p3 "${MINIMUM_AUC_GAIN_VS_P3:-0.05}" \
    --minimum_precision "${MINIMUM_PRECISION:-0.55}" \
    --minimum_rescued "${MINIMUM_RESCUED:-10}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
    --minimum_positive_events "${MINIMUM_POSITIVE_EVENTS:-3}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2; exit 2 ;;
esac
