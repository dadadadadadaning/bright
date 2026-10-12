#!/usr/bin/env bash
set -euo pipefail

# S9-XHPI: frozen S8 descriptor -> metric -> prototype router -> instance cache
# -> retrieval validation -> safe pixel validation -> fixed test once.

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S9_XHPI/BRIGHT/S9_XHPI_K4_C64}"

RAW_TRAIN="${RAW_TRAIN:-${OUTPUT_ROOT}/raw_train_gt.pt}"
RAW_VAL="${RAW_VAL:-${OUTPUT_ROOT}/raw_val_pred.pt}"
RAW_TEST="${RAW_TEST:-${OUTPUT_ROOT}/raw_test_pred.pt}"
PROJECTOR_CKPT="${PROJECTOR_CKPT:-${OUTPUT_ROOT}/metric_projector.pth}"
MEMORY_PATH="${MEMORY_PATH:-${OUTPUT_ROOT}/hierarchical_memory.pt}"
RETRIEVAL_LOCK="${RETRIEVAL_LOCK:-${OUTPUT_ROOT}/best_val_config.json}"
PIXEL_LOCK="${PIXEL_LOCK:-${OUTPUT_ROOT}/best_pixel_config.json}"

PROTOTYPES_PER_CLASS="${PROTOTYPES_PER_CLASS:-4}"
MAX_PER_EVENT_CLASS="${MAX_PER_EVENT_CLASS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

require_s8() {
  : "${S8_CKPT:?Set S8_CKPT to a complete TRAINED S8 checkpoint (not an untrained step-0 head)}"
  test -s "${S8_CKPT}"
}

preflight() {
  python -m py_compile \
    model/DamageFormer_S9_XHPI.py \
    script/standard_ML/s9_xhpi_common.py \
    script/standard_ML/extract_S9_XHPI_features.py \
    script/standard_ML/train_S9_XHPI_metric.py \
    script/standard_ML/build_S9_XHPI_memory.py \
    script/standard_ML/eval_S9_XHPI_retrieval.py \
    script/standard_ML/eval_S9_X2B_prototype_guard.py \
    script/standard_ML/eval_S9_XHPI_online.py
  python -m unittest tests/test_s9_xhpi.py -v
}

extract_split() {
  local split="$1"
  local list="$2"
  local source="$3"
  local output="$4"
  require_s8
  python -u script/standard_ML/extract_S9_XHPI_features.py \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${list}" \
    --checkpoint "${S8_CKPT}" \
    --output "${output}" \
    --split "${split}" \
    --region_source "${source}" \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS}" \
    --image_size 1024 \
    --min_instance_area 32 \
    --max_instance_area 300000 \
    --min_gt_coverage 0.50 \
    --min_label_purity 0.70 \
    --max_local_tokens 16
}

train_metric() {
  test -s "${RAW_TRAIN}"
  python -u script/standard_ML/train_S9_XHPI_metric.py \
    --raw_train "${RAW_TRAIN}" \
    --output "${PROJECTOR_CKPT}" \
    --history "${OUTPUT_ROOT}/metric_training_history.json" \
    --device cuda:0 \
    --embed_dim 128 \
    --hidden_dim 256 \
    --epochs 30 \
    --batch_size 256 \
    --learning_rate 2e-4 \
    --dev_event_fraction 0.20
}

build_memory() {
  test -s "${RAW_TRAIN}"
  test -s "${PROJECTOR_CKPT}"
  python -u script/standard_ML/build_S9_XHPI_memory.py \
    --raw_train "${RAW_TRAIN}" \
    --projector "${PROJECTOR_CKPT}" \
    --output "${MEMORY_PATH}" \
    --report "${OUTPUT_ROOT}/memory_report.json" \
    --device cuda:0 \
    --prototypes_per_class "${PROTOTYPES_PER_CLASS}" \
    --max_per_event_class "${MAX_PER_EVENT_CLASS}" \
    --hard_fraction 0.20
}

tune_retrieval() {
  test -s "${RAW_VAL}"; test -s "${PROJECTOR_CKPT}"; test -s "${MEMORY_PATH}"
  python -u script/standard_ML/eval_S9_XHPI_retrieval.py \
    --queries "${RAW_VAL}" \
    --projector "${PROJECTOR_CKPT}" \
    --memory "${MEMORY_PATH}" \
    --output "${OUTPUT_ROOT}/val_retrieval_results.json" \
    --mode val \
    --device cuda:0
}

tune_pixel() {
  require_s8
  test -s "${RETRIEVAL_LOCK}"
  python -u script/standard_ML/eval_S9_XHPI_online.py \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${VAL_LIST}" \
    --s8_checkpoint "${S8_CKPT}" \
    --projector "${PROJECTOR_CKPT}" \
    --memory "${MEMORY_PATH}" \
    --retrieval_config "${RETRIEVAL_LOCK}" \
    --output "${OUTPUT_ROOT}/val_pixel_results.json" \
    --mode val \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS}"
}

prototype_guard() {
  test -s "${RAW_VAL}"; test -s "${PROJECTOR_CKPT}"; test -s "${MEMORY_PATH}"
  python -u script/standard_ML/eval_S9_X2B_prototype_guard.py \
    --queries "${RAW_VAL}" \
    --projector "${PROJECTOR_CKPT}" \
    --memory "${MEMORY_PATH}" \
    --output "${OUTPUT_ROOT}/val_prototype_guard.json" \
    --device cuda:0 \
    --batch_size 1024 \
    --prototype_beta 8.0
}

test_retrieval() {
  test -s "${RAW_TEST}"; test -s "${RETRIEVAL_LOCK}"
  python -u script/standard_ML/eval_S9_XHPI_retrieval.py \
    --queries "${RAW_TEST}" \
    --projector "${PROJECTOR_CKPT}" \
    --memory "${MEMORY_PATH}" \
    --output "${OUTPUT_ROOT}/test_retrieval_once.json" \
    --mode test \
    --locked_config "${RETRIEVAL_LOCK}" \
    --device cuda:0
}

test_pixel() {
  require_s8
  test -s "${PIXEL_LOCK}"
  python -u script/standard_ML/eval_S9_XHPI_online.py \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TEST_LIST}" \
    --s8_checkpoint "${S8_CKPT}" \
    --projector "${PROJECTOR_CKPT}" \
    --memory "${MEMORY_PATH}" \
    --retrieval_config "${RETRIEVAL_LOCK}" \
    --pixel_config "${PIXEL_LOCK}" \
    --output "${OUTPUT_ROOT}/test_pixel_once.json" \
    --mode test \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS}"
}

case "${MODE}" in
  preflight) preflight ;;
  extract_train) extract_split train "${TRAIN_LIST}" gt "${RAW_TRAIN}" ;;
  train_metric) train_metric ;;
  build_memory) build_memory ;;
  extract_val) extract_split val "${VAL_LIST}" pred "${RAW_VAL}" ;;
  tune_retrieval) tune_retrieval ;;
  prototype_guard) prototype_guard ;;
  tune_pixel) tune_pixel ;;
  extract_test) extract_split test "${TEST_LIST}" pred "${RAW_TEST}" ;;
  test_retrieval) test_retrieval ;;
  test_pixel) test_pixel ;;
  all_train_val)
    preflight
    extract_split train "${TRAIN_LIST}" gt "${RAW_TRAIN}"
    train_metric
    build_memory
    extract_split val "${VAL_LIST}" pred "${RAW_VAL}"
    tune_retrieval
    tune_pixel
    ;;
  fixed_test_once)
    extract_split test "${TEST_LIST}" pred "${RAW_TEST}"
    test_retrieval
    test_pixel
    ;;
  *)
    echo "Unknown mode '${MODE}'" >&2
    echo "Expected preflight|extract_train|train_metric|build_memory|extract_val|tune_retrieval|prototype_guard|tune_pixel|extract_test|test_retrieval|test_pixel|all_train_val|fixed_test_once" >&2
    exit 2
    ;;
esac
