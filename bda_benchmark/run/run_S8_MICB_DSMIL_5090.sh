#!/usr/bin/env bash
set -euo pipefail

# S8/P3 launch script for one RTX 5090/4090-class GPU.
#
# Usage:
#   bash run/run_S8_MICB_DSMIL_5090.sh preflight
#   BASE_CKPT=/abs/best_val_miou.pth bash run/run_S8_MICB_DSMIL_5090.sh pilot
#   BASE_CKPT=/abs/best_val_miou.pth bash run/run_S8_MICB_DSMIL_5090.sh full
#   RESUME_CKPT=/abs/last.pth BASE_CKPT=/abs/base.pth \
#     bash run/run_S8_MICB_DSMIL_5090.sh full

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16}"
EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU:-68.6493}"
BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE:-0.10}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    model/DamageFormer_S8_MICB_DSMIL.py \
    script/standard_ML/train_S8_MICB_DSMIL.py
  python -m unittest tests/test_s8_micb_dsmil.py -v
}

train_run() {
  local steps="$1"
  local max_iters="$2"
  local eval_every="$3"
  local head_only_steps="$4"
  : "${BASE_CKPT:?Set BASE_CKPT to the exact S3V2 best_val_miou.pth}"
  test -s "${BASE_CKPT}"
  mkdir -p "${OUTPUT_ROOT}"

  local resume_args=()
  if [[ -n "${RESUME_CKPT:-}" ]]; then
    test -s "${RESUME_CKPT}"
    resume_args+=(--resume "${RESUME_CKPT}" --skip_initial_eval)
  fi
  local region_args=()
  if [[ "${USE_ASYMMETRIC_REGIONS:-0}" == "1" ]]; then
    region_args+=(--use_asymmetric_regions)
  fi

  python -u script/standard_ML/train_S8_MICB_DSMIL.py \
    --dataset BRIGHT \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --test_dataset_path "${DATASET_ROOT}" \
    --test_data_list_path "${TEST_LIST}" \
    --base_ckpt "${BASE_CKPT}" \
    --output_root "${OUTPUT_ROOT}" \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --accumulation_steps 4 \
    --optimizer_steps "${steps}" \
    --max_iters "${max_iters}" \
    --crop_size 640 \
    --eval_every "${eval_every}" \
    --log_every 20 \
    --num_workers 8 \
    --amp \
    --roi_alpha 0.1 \
    --token_dim 128 \
    --mil_topk 8 \
    --max_tokens_per_bag 256 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
    --bag_diagnostic_limit 2000 \
    --mil_residual_scale 0.0 \
    --base_learning_rate 1e-5 \
    --head_learning_rate 1e-4 \
    --head_only_steps "${head_only_steps}" \
    --mil_warmup_steps 1000 \
    --lambda_mil 0.25 \
    --lambda_exist_aux 0.25 \
    --lambda_severity_aux 0.25 \
    --lambda_intact_guard 0.05 \
    --lambda_coverage 0.0 \
    --expected_base_val_miou "${EXPECTED_BASE_VAL_MIOU}" \
    --base_miou_tolerance "${BASE_MIOU_TOLERANCE}" \
    "${resume_args[@]}" \
    "${region_args[@]}"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  smoke)
    train_run 20 320 10 5
    ;;
  pilot)
    train_run 10000 160000 500 500
    ;;
  full)
    train_run 50000 800000 500 500
    ;;
  *)
    echo "Unknown mode '${MODE}'; expected preflight|smoke|pilot|full" >&2
    exit 2
    ;;
esac
