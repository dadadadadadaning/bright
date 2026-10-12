#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BASE_CKPT=/absolute/path/best_val_miou.pth bash run/run_S7_SIRM_5090.sh build
#   BASE_CKPT=/absolute/path/best_val_miou.pth bash run/run_S7_SIRM_5090.sh tune
#   BASE_CKPT=/absolute/path/best_val_miou.pth bash run/run_S7_SIRM_5090.sh test
#   BASE_CKPT=/absolute/path/best_val_miou.pth bash run/run_S7_SIRM_5090.sh all
#
# BASE_CKPT may be the S3V2 K=3 checkpoint.  Prototype parameters are removed
# and the remaining weights reproduce the exact prototype-bypass branch.

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/..:${PROJECT_ROOT}:${PYTHONPATH:-}"

: "${BASE_CKPT:?Set BASE_CKPT to the exact S1/S3 checkpoint used for M0/M2 comparison}"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S7_SIRM/BRIGHT/S7_SIRM_static_cache}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_ROOT}/train_instance_cache.pt}"
VAL_JSON="${VAL_JSON:-${OUTPUT_ROOT}/val_grid_results.json}"
LOCKED_CONFIG="${LOCKED_CONFIG:-${OUTPUT_ROOT}/best_val_config.json}"
TEST_JSON="${TEST_JSON:-${OUTPUT_ROOT}/test_results.json}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

build_cache() {
  python -u script/standard_ML/build_S7_SIRM_cache.py \
    --dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --checkpoint "${BASE_CKPT}" \
    --output_cache "${CACHE_PATH}" \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers 4 \
    --image_size 1024 \
    --roi_alpha 0.1 \
    --min_instance_area 32 \
    --min_label_purity 0.90 \
    --max_per_event_class 64 \
    --max_per_class 2048 \
    --hard_fraction 0.25
}

tune_val() {
  test -f "${CACHE_PATH}"
  python -u script/standard_ML/eval_S7_SIRM.py \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${VAL_LIST}" \
    --split_name val \
    --checkpoint "${BASE_CKPT}" \
    --cache "${CACHE_PATH}" \
    --output_json "${VAL_JSON}" \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers 4 \
    --roi_alpha 0.1 \
    --min_instance_area 32 \
    --search_val \
    --min_accept_miou_gain 0.10 \
    --max_minority_iou_drop 0.20
}

test_locked() {
  test -f "${CACHE_PATH}"
  test -f "${LOCKED_CONFIG}"
  python -u script/standard_ML/eval_S7_SIRM.py \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TEST_LIST}" \
    --split_name test \
    --checkpoint "${BASE_CKPT}" \
    --cache "${CACHE_PATH}" \
    --config_json "${LOCKED_CONFIG}" \
    --output_json "${TEST_JSON}" \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers 4 \
    --roi_alpha 0.1 \
    --min_instance_area 32
}

case "${MODE}" in
  build)
    build_cache
    ;;
  tune)
    tune_val
    ;;
  test)
    test_locked
    ;;
  all)
    build_cache
    tune_val
    test_locked
    ;;
  *)
    echo "Unknown mode: ${MODE}; expected build|tune|test|all" >&2
    exit 2
    ;;
esac
