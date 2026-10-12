#!/usr/bin/env bash
set -euo pipefail

# Controlled S8-R0...R5 ablations on one RTX 5090.
#
# Examples:
#   bash run/run_S8_reliable_cross_modal_5090.sh preflight
#   VARIANT=r0 bash run/run_S8_reliable_cross_modal_5090.sh baseline
#   VARIANT=r1 bash run/run_S8_reliable_cross_modal_5090.sh smoke
#   VARIANT=r1 bash run/run_S8_reliable_cross_modal_5090.sh pilot
#
# For a strict ablation, keep BASE_CKPT identical for R1...R5.  Loading the
# previous R checkpoint is supported as a warm start, but is a different
# sequential-training protocol and must not be mixed into the same table.

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="${VARIANT:-r1}"
GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_ReliableCrossModal}"
EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU:-68.6493}"
BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE:-0.5}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  bash -n \
    run/run_S8_reliable_cross_modal_5090.sh \
    run/run_S8_reliable_cross_modal_sequence.sh
  python -m py_compile \
    model/reliable_cross_modal/blocks.py \
    model/DamageFormer_S8_ReliableCrossModal.py \
    script/standard_ML/train_S5_S8_GPU.py \
    script/standard_ML/train_S8_reliable_cross_modal.py
  python -m unittest tests/test_s8_reliable_cross_modal.py -v
}

train_run() {
  local optimizer_steps="$1"
  local eval_every="$2"
  local val_max_batches="$3"
  test -s "${BASE_CKPT}"
  test -s "${TRAIN_LIST}"
  test -s "${VAL_LIST}"
  mkdir -p "${OUTPUT_ROOT}"

  local extra_args=()
  if [[ -n "${RESUME_CKPT:-}" ]]; then
    test -s "${RESUME_CKPT}"
    extra_args+=(--resume "${RESUME_CKPT}")
  fi
  if [[ -n "${EXP_NAME:-}" ]]; then
    extra_args+=(--exp_name "${EXP_NAME}")
  fi

  python -u script/standard_ML/train_S8_reliable_cross_modal.py \
    --variant "${VARIANT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --base_ckpt "${BASE_CKPT}" \
    --output_root "${OUTPUT_ROOT}" \
    --image_size 640 \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --accumulation_steps 8 \
    --optimizer_steps "${optimizer_steps}" \
    --num_workers 8 \
    --amp_dtype bf16 \
    --base_learning_rate 1e-5 \
    --new_learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --head_only_steps 500 \
    --relation_projection_channels 32 \
    --relation_channels 64 \
    --gate_alpha_max 0.1 \
    --rectifier_alpha_max 0.05 \
    --roi_alpha 0.1 \
    --lambda_change 0.1 \
    --lambda_exist 0.1 \
    --lambda_severity 0.1 \
    --aux_warmup_steps 0 \
    --aux_ramp_steps 500 \
    --eval_every "${eval_every}" \
    --log_every 20 \
    --save_every 1000 \
    --val_max_batches "${val_max_batches}" \
    --expected_base_val_miou "${EXPECTED_BASE_VAL_MIOU}" \
    --base_miou_tolerance "${BASE_MIOU_TOLERANCE}" \
    --seed 3407 \
    "${extra_args[@]}"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  baseline)
    # Full validation, zero optimizer steps.  The trainer's initial-baseline
    # guard aborts if checkpoint/model semantics do not match.
    train_run 0 500 0
    ;;
  smoke)
    train_run 20 10 10
    ;;
  pilot)
    train_run 3000 500 0
    ;;
  full)
    train_run 10000 500 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|baseline|smoke|pilot|full" >&2
    exit 2
    ;;
esac
