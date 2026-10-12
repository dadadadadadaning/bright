#!/usr/bin/env bash
set -euo pipefail

# CMR-P0 is a frozen validation-only pooling audit.  It never trains P3,
# changes predictions, builds from test data, or admits a key directly to test.
MODE="${1:-audit}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${OUTPUT_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
QUERY_CACHE="${QUERY_CACHE:-${OUTPUT_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT="${RESULT:-${OUTPUT_ROOT}/CMR_P0_pooling_audit_val_seed${SEED}.json}"

check_protocol() {
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  for path in "${TRAIN_LIST}" "${VAL_LIST}"; do
    test -s "${path}" || { echo "Missing split list: ${path}" >&2; exit 3; }
  done
  local observed
  observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
  [[ "${observed}" == "${EXPECTED_P3_SHA256}" ]] || {
    echo "P3 SHA256 mismatch: ${observed}" >&2
    echo "Expected: ${EXPECTED_P3_SHA256}" >&2
    exit 4
  }
}

preflight() {
  python -m py_compile \
    model/cmr/pooling.py \
    script/standard_ML/build_CMR_pooling_cache.py \
    script/standard_ML/audit_CMR_pooling.py
  python -m unittest tests.test_cmr_pooling -v
  if [[ -f "${P3_CKPT}" ]]; then check_protocol; fi
  echo "CMR-P0 pooling preflight passed."
}

build_one() {
  local role="$1"
  local list="$2"
  local output="$3"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Cache already exists: ${output}"
    return 0
  fi
  mkdir -p "$(dirname "${output}")"
  python -u script/standard_ML/build_CMR_pooling_cache.py \
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
    --capacity_per_event_class "${CAPACITY_PER_EVENT_CLASS:-200}" \
    --seed "${SEED}"
}

build_caches() {
  check_protocol
  build_one train "${TRAIN_LIST}" "${TRAIN_CACHE}"
  build_one query "${VAL_LIST}" "${QUERY_CACHE}"
}

run_audit() {
  build_caches
  mkdir -p "$(dirname "${RESULT}")"
  python -u script/standard_ML/audit_CMR_pooling.py \
    --train_cache "${TRAIN_CACHE}" \
    --query_cache "${QUERY_CACHE}" \
    --output "${RESULT}" \
    --device cuda:0 \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}"
}

case "${MODE}" in
  preflight) preflight ;;
  build) build_caches ;;
  audit) run_audit ;;
  all)
    preflight
    run_audit
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|build|audit|all" >&2
    exit 2
    ;;
esac
