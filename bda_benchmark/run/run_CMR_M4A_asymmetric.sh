#!/usr/bin/env bash
set -euo pipefail

# No-training rescue audit. Raw Mean remains the primary ranking. Only a
# Train-locked, high-confidence Raw Non-D -> M3T D promotion is permitted.
MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
M4_ROOT="${M4_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M4A_ASYMMETRIC}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${RESULT_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${RESULT_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
M3T_ROOT="${M3T_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M3T_TOP1/CMR_M3T_top20_epochs6_seed${SEED}}"
CHECKPOINT="${CHECKPOINT:-${M3T_ROOT}/final_top1_pairwise.pth}"
OUTPUT="${OUTPUT:-${M4_ROOT}/CMR_M4A_train_locked_asymmetric_val_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/cmr_asymmetric_rescue_utils.py \
    script/standard_ML/audit_CMR_M4A_asymmetric.py
  python -m unittest \
    tests.test_cmr_m3_pairwise \
    tests.test_cmr_m3c_conditional \
    tests.test_cmr_m3t_top1 \
    tests.test_cmr_m4a_asymmetric -v
  echo "CMR-M4A asymmetric rescue preflight passed."
}

audit_rescue() {
  test -s "${TRAIN_CANDIDATES}" || { echo "Missing Train candidates" >&2; exit 3; }
  test -s "${VAL_CANDIDATES}" || { echo "Missing Validation candidates" >&2; exit 3; }
  test -s "${CHECKPOINT}" || { echo "Missing frozen M3T checkpoint: ${CHECKPOINT}" >&2; exit 3; }
  mkdir -p "$(dirname "${OUTPUT}")"
  python -u script/standard_ML/audit_CMR_M4A_asymmetric.py \
    --train_candidates "${TRAIN_CANDIDATES}" \
    --val_candidates "${VAL_CANDIDATES}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --device cuda:0 \
    --thresholds "${RESCUE_THRESHOLDS:-0,0.0025,0.005,0.01,0.02,0.03,0.05}" \
    --minimum_rescued "${MINIMUM_RESCUED:-10}" \
    --minimum_precision "${MINIMUM_PRECISION:-0.55}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
    --minimum_positive_events "${MINIMUM_POSITIVE_EVENTS:-3}" \
    --maximum_destroyed_drop "${MAXIMUM_DESTROYED_DROP:-0.01}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit_rescue ;;
  all)
    preflight
    audit_rescue
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2
    exit 2
    ;;
esac

