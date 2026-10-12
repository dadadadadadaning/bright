#!/usr/bin/env bash
set -euo pipefail

# CMR-H0 uses Train only.  It mines fixed hard pairs for later P3 training;
# it never reads Validation/Test features and never changes a prediction.
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
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
H0_ROOT="${H0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H0_TRAIN_ONLY}"
PAIRS_OUTPUT="${PAIRS_OUTPUT:-${H0_ROOT}/CMR_H0_train_hard_pairs_seed${SEED}.pth}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-${H0_ROOT}/CMR_H0_train_hard_pair_audit_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/cmr_hard_mining_utils.py \
    script/standard_ML/audit_CMR_H0_train_only.py
  python -m unittest \
    tests.test_cmr_h0_train_only -v
  echo "CMR-H0 Train-only hard-mining preflight passed."
}

build_train_cache_if_missing() {
  if [[ -s "${TRAIN_CACHE}" ]]; then
    echo "Train cache already exists: ${TRAIN_CACHE}"
    return 0
  fi
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  test -s "${TRAIN_LIST}" || { echo "Missing Train list: ${TRAIN_LIST}" >&2; exit 3; }
  test -s "${VAL_LIST}" || { echo "Missing Validation list for split-overlap guard: ${VAL_LIST}" >&2; exit 3; }
  mkdir -p "$(dirname "${TRAIN_CACHE}")"
  python -u script/standard_ML/build_CMR_pooling_cache.py \
    --role train \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TRAIN_LIST}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_data_list_path "${VAL_LIST}" \
    --p3_checkpoint "${P3_CKPT}" \
    --output "${TRAIN_CACHE}" \
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

audit_pairs() {
  build_train_cache_if_missing
  mkdir -p "${H0_ROOT}"
  python -u script/standard_ML/audit_CMR_H0_train_only.py \
    --train_cache "${TRAIN_CACHE}" \
    --pairs_output "${PAIRS_OUTPUT}" \
    --audit_output "${AUDIT_OUTPUT}" \
    --device cuda:0 \
    --positive_topk "${POSITIVE_TOPK:-5}" \
    --negative_topk "${NEGATIVE_TOPK:-10}" \
    --positive_max_per_event "${POSITIVE_MAX_PER_EVENT:-1}" \
    --negative_max_per_event "${NEGATIVE_MAX_PER_EVENT:-2}" \
    --chunk_size "${QUERY_CHUNK:-256}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit_pairs ;;
  all)
    preflight
    audit_pairs
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2
    exit 2
    ;;
esac

