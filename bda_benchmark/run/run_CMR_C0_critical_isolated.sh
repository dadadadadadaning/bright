#!/usr/bin/env bash
set -euo pipefail

# CMR-C0 reads P3/P0/M3/V1 region artifacts and writes only its own directory.
MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CMR_C0_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CMR_C0_ENV}"
fi

SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
M3_ROOT="${M3_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
V1_REGION_ROOT="${V1_REGION_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V1_ISOLATED}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_C0_CRITICAL_ISOLATED}"

TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${M3_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${M3_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
SUPPORT_REGION="${SUPPORT_REGION:-${V1_REGION_ROOT}/CMR_V1_REGION_train_support_seed${SEED}.pth}"
TRAIN_QUERY_REGION="${TRAIN_QUERY_REGION:-${V1_REGION_ROOT}/CMR_V1_REGION_train_query_seed${SEED}.pth}"
VAL_QUERY_REGION="${VAL_QUERY_REGION:-${V1_REGION_ROOT}/CMR_V1_REGION_val_query_seed${SEED}.pth}"

SUPPORT="${SUPPORT:-${RESULT_ROOT}/CMR_C0_train_support_cases_seed${SEED}.pth}"
TRAIN_QUERY="${TRAIN_QUERY:-${RESULT_ROOT}/CMR_C0_train_query_cases_seed${SEED}.pth}"
VAL_QUERY="${VAL_QUERY:-${RESULT_ROOT}/CMR_C0_val_query_cases_seed${SEED}.pth}"
AUDIT_RESULT="${AUDIT_RESULT:-${RESULT_ROOT}/CMR_C0_critical_audit_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    model/cmr_c0/case_card.py \
    script/standard_ML/cmr_c0/build_case_cache.py \
    script/standard_ML/cmr_c0/audit_critical_verifier.py
  python -m unittest \
    tests.cmr_c0.test_case_card tests.cmr_c0.test_retrieval \
    tests.cmr_c0.test_audit_pipeline -v
  echo "CMR-C0 isolated preflight passed."
}

require_inputs() {
  for path in \
    "${TRAIN_P0}" "${VAL_P0}" "${TRAIN_CANDIDATES}" "${VAL_CANDIDATES}" \
    "${SUPPORT_REGION}" "${TRAIN_QUERY_REGION}" "${VAL_QUERY_REGION}" "${P3_CKPT}"; do
    test -s "${path}" || { echo "Missing read-only upstream input: ${path}" >&2; exit 3; }
  done
}

build_one() {
  local role="$1" p0="$2" candidates="$3" region="$4" output="$5"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Frozen CMR-C0 case cache already exists: ${output}"
    return
  fi
  python -u script/standard_ML/cmr_c0/build_case_cache.py \
    --role "${role}" --dataset_path "${DATASET_ROOT}" \
    --p0_cache "${p0}" --candidate_cache "${candidates}" \
    --region_cache "${region}" --p3_checkpoint "${P3_CKPT}" \
    --output "${output}" --device cuda:0 --num_workers "${NUM_WORKERS:-8}" \
    --mask_threshold "${MASK_THRESHOLD:-0.05}" \
    --critical_min "${CRITICAL_MIN:-4}" --critical_max "${CRITICAL_MAX:-32}" \
    --nms_radius "${NMS_RADIUS:-1.5}"
}

build_cases() {
  require_inputs
  mkdir -p "${RESULT_ROOT}"
  build_one support "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${SUPPORT_REGION}" "${SUPPORT}"
  build_one train_query "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${TRAIN_QUERY_REGION}" "${TRAIN_QUERY}"
  build_one validation_query "${VAL_P0}" "${VAL_CANDIDATES}" "${VAL_QUERY_REGION}" "${VAL_QUERY}"
}

audit() {
  build_cases
  python -u script/standard_ML/cmr_c0/audit_critical_verifier.py \
    --support "${SUPPORT}" --train_queries "${TRAIN_QUERY}" \
    --validation_queries "${VAL_QUERY}" --output "${AUDIT_RESULT}" \
    --device cuda:0 --neighbor_k "${NEIGHBOR_K:-3}" \
    --class_temperature "${CLASS_TEMPERATURE:-0.10}" \
    --logistic_l2 "${LOGISTIC_L2:-0.01}"
}

case "${MODE}" in
  preflight) preflight ;;
  build) build_cases ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|build|audit|all" >&2; exit 2 ;;
esac
