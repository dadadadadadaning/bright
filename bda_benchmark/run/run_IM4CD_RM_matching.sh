#!/usr/bin/env bash
set -euo pipefail

# Validation-only transfer of IM4CD's hard local response-peak idea into the
# existing class x event balanced RM region protocol.  This is not a full
# IM4CD/CRF reproduction and never changes frozen P3 predictions.

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/IM4CD_RM_MATCHING}"
MEMORY="${MEMORY:-${OUTPUT_ROOT}/IM0_train_memory_seed${SEED}.pth}"
QUERIES="${QUERIES:-${OUTPUT_ROOT}/IM0_val_queries_seed${SEED}.pth}"
CALIBRATION_QUERIES="${CALIBRATION_QUERIES:-${OUTPUT_ROOT}/IM1D_natural_train_queries_seed${SEED}.pth}"
RESULT="${RESULT:-${OUTPUT_ROOT}/IM0_hard_peak_region_audit_seed${SEED}.json}"
IM1_RESULT="${IM1_RESULT:-${OUTPUT_ROOT}/IM1_matching_reliability_val_seed${SEED}.json}"
IM1D_RESULT="${IM1D_RESULT:-${OUTPUT_ROOT}/IM1D_natural_crossfit_reliability_val_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/im4cd_matching_runtime.py \
    script/standard_ML/im1_matching_reliability_runtime.py \
    script/standard_ML/build_RM0_region_cache.py \
    script/standard_ML/audit_IM4CD_region_matching.py \
    script/standard_ML/audit_IM1_matching_reliability.py \
    script/standard_ML/audit_IM1D_natural_crossfit_reliability.py
  python -m unittest \
    tests.test_im4cd_region_matching \
    tests.test_im1_matching_reliability \
    tests.test_rm_region_retrieval -v
  echo "IM4CD-RM matching preflight passed."
}

build_cache() {
  local role="$1"
  local list="$2"
  local output="$3"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Cache already exists: ${output}"
    return 0
  fi
  python -u script/standard_ML/build_RM0_region_cache.py \
    --role "${role}" \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${list}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_data_list_path "${VAL_LIST}" \
    --p3_checkpoint "${P3_CKPT}" \
    --output "${output}" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-8}" \
    --min_area "${MIN_AREA:-32}" \
    --max_area "${MAX_AREA:-300000}" \
    --min_building_coverage "${MIN_BUILDING_COVERAGE:-0.50}" \
    --min_class_purity "${MIN_CLASS_PURITY:-0.75}" \
    --capacity_per_class "${CAPACITY_PER_CLASS:-2000}" \
    --balance_mode class_event \
    --capacity_per_event_class "${CAPACITY_PER_EVENT_CLASS:-200}" \
    --matching_stats \
    --matching_window "${MATCHING_WINDOW:-5}" \
    --matching_temperature "${MATCHING_TEMPERATURE:-0.10}" \
    --seed "${SEED}"
}

build_all() {
  test -s "${P3_CKPT}" || { echo "Missing original P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  local observed
  observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
  [[ "${observed}" == "${EXPECTED_P3_SHA256}" ]] || {
    echo "P3 SHA256 mismatch: ${observed}" >&2
    echo "Expected: ${EXPECTED_P3_SHA256}" >&2
    exit 4
  }
  mkdir -p "${OUTPUT_ROOT}"
  build_cache memory "${TRAIN_LIST}" "${MEMORY}"
  build_cache query "${VAL_LIST}" "${QUERIES}"
}

audit() {
  build_all
  python -u script/standard_ML/audit_IM4CD_region_matching.py \
    --memory "${MEMORY}" \
    --queries "${QUERIES}" \
    --output "${RESULT}" \
    --device cuda:0 \
    --weights "${MATCHING_WEIGHTS:-0,0.25,0.5,1.0}" \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}"
}

reliability() {
  build_all
  python -u script/standard_ML/audit_IM1_matching_reliability.py \
    --memory "${MEMORY}" \
    --queries "${QUERIES}" \
    --output "${IM1_RESULT}" \
    --device cuda:0 \
    --raw_weight "${IM1_RAW_WEIGHT:-1.0}" \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --coverages "${IM1_COVERAGES:-0.01,0.025,0.05,0.10,0.20}" \
    --logistic_l2 "${IM1_LOGISTIC_L2:-0.01}"
}

reliability_d() {
  build_all
  build_cache calibration "${TRAIN_LIST}" "${CALIBRATION_QUERIES}"
  python -u script/standard_ML/audit_IM1D_natural_crossfit_reliability.py \
    --memory "${MEMORY}" \
    --calibration_queries "${CALIBRATION_QUERIES}" \
    --validation_queries "${QUERIES}" \
    --output "${IM1D_RESULT}" \
    --device cuda:0 \
    --raw_weight "${IM1_RAW_WEIGHT:-1.0}" \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --coverages "${IM1_COVERAGES:-0.01,0.025,0.05,0.10,0.20}" \
    --logistic_l2 "${IM1_LOGISTIC_L2:-0.01}"
}

case "${MODE}" in
  preflight) preflight ;;
  build) build_all ;;
  audit) audit ;;
  reliability) reliability ;;
  reliability_d) reliability_d ;;
  im1)
    preflight
    reliability
    ;;
  im1d)
    preflight
    reliability_d
    ;;
  all)
    preflight
    audit
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|build|audit|reliability|reliability_d|im1|im1d|all" >&2
    exit 2
    ;;
esac
