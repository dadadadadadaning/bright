#!/usr/bin/env bash
set -euo pipefail

# CMR-V2 isolated EC0--EC3 audit.  Reads frozen P0/M3 only; never fuses P3.
MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CMR_V2_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CMR_V2_ENV}"
fi

SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
M3_ROOT="${M3_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V2_EVENT_FACTORIZED}"
TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${M3_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${M3_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
OUTPUT="${OUTPUT:-${RESULT_ROOT}/CMR_V2_EC0_EC3_seed${SEED}.json}"
DEVICE="${DEVICE:-cuda:0}"

preflight() {
  python -m py_compile \
    model/cmr_v2/event_factorization.py \
    script/standard_ML/cmr_v2/audit_ec0_ec3.py
  python -m unittest tests.cmr_v2.test_event_factorization -v
  echo "CMR-V2 EC0--EC3 isolated preflight passed."
}

audit() {
  for path in "${TRAIN_P0}" "${VAL_P0}" "${TRAIN_CANDIDATES}" "${VAL_CANDIDATES}"; do
    test -s "${path}" || { echo "Missing read-only upstream input: ${path}" >&2; exit 3; }
  done
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/cmr_v2/audit_ec0_ec3.py \
    --train_p0 "${TRAIN_P0}" --val_p0 "${VAL_P0}" \
    --train_candidates "${TRAIN_CANDIDATES}" --val_candidates "${VAL_CANDIDATES}" \
    --output "${OUTPUT}" --device "${DEVICE}" \
    --case_k "${CASE_K:-5}" --temperature "${TEMPERATURE:-0.10}" \
    --scale_floor "${SCALE_FLOOR:-0.001}" --query_chunk "${QUERY_CHUNK:-2048}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2; exit 2 ;;
esac
