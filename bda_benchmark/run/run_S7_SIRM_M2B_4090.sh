#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BASE_CKPT=/absolute/path/best_val_miou.pth \
#   CACHE_PATH=/absolute/path/train_instance_cache.pt \
#     bash run/run_S7_SIRM_M2B_4090.sh tune
#   BASE_CKPT=... CACHE_PATH=... bash run/run_S7_SIRM_M2B_4090.sh test
#   BASE_CKPT=... CACHE_PATH=... bash run/run_S7_SIRM_M2B_4090.sh all
#
# M2b deliberately reuses M2a's exact train-only cache.  This isolates the
# directional/base-uncertainty gate and preserves M2a as a completed ablation.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S7_SIRM/BRIGHT/S7_SIRM_M2B_directional_uncertainty}"
CACHE_PATH="${CACHE_PATH:-/root/shared-nvme/BRIGHT-checkpoints/S7_SIRM/BRIGHT/S7_SIRM_K3_direct256_bestval/train_instance_cache.pt}"
VAL_JSON="${VAL_JSON:-${OUTPUT_ROOT}/val_grid_results.json}"
LOCKED_CONFIG="${LOCKED_CONFIG:-${OUTPUT_ROOT}/best_val_config.json}"
TEST_JSON="${TEST_JSON:-${OUTPUT_ROOT}/test_results.json}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

tune_val() {
  test -f "${CACHE_PATH}"
  python -u script/standard_ML/eval_S7_SIRM_M2B.py \
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
  python -u script/standard_ML/eval_S7_SIRM_M2B.py \
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

validation_accepted() {
  python - "${LOCKED_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("accepted") is True else 1)
PY
}

case "${MODE}" in
  tune)
    tune_val
    ;;
  test)
    test_locked
    ;;
  all)
    tune_val
    if validation_accepted; then
      test_locked
    else
      echo "M2b validation rejected: locked active test skipped." >&2
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}; expected tune|test|all" >&2
    exit 2
    ;;
esac
