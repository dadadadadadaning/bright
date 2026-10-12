#!/usr/bin/env bash
set -euo pipefail

# S9-R1: train-only CEAM -> balanced validation reference -> pure metric gate.
# This launcher has no Test mode and never builds prototypes or performs fusion.

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
S9_OUTPUT_ROOT="${S9_OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S9_XHPI/BRIGHT/S9_XHPI_K4_C64_S8_bestval}"
R1_OUTPUT_ROOT="${R1_OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S9_R1_CEAM/BRIGHT/S9_R1_ChangeLocal_Angular}"

RAW_TRAIN="${RAW_TRAIN:-${S9_OUTPUT_ROOT}/raw_train_gt.pt}"
RAW_VAL_PRED="${RAW_VAL_PRED:-${S9_OUTPUT_ROOT}/raw_val_pred.pt}"
R0_PAIRED_VAL="${R0_PAIRED_VAL:-${S9_OUTPUT_ROOT}/raw_val_gt_pred_pairs.pt}"
CEAM_CKPT="${CEAM_CKPT:-${R1_OUTPUT_ROOT}/best_ceam.pth}"
CEAM_HISTORY="${CEAM_HISTORY:-${R1_OUTPUT_ROOT}/training_history.json}"
REFERENCE_PATH="${REFERENCE_PATH:-${R1_OUTPUT_ROOT}/validation_reference.pt}"
REFERENCE_REPORT="${REFERENCE_REPORT:-${R1_OUTPUT_ROOT}/reference_report.json}"
VALIDATION_REPORT="${VALIDATION_REPORT:-${R1_OUTPUT_ROOT}/validation_results.json}"
METRIC_GATE="${METRIC_GATE:-${R1_OUTPUT_ROOT}/metric_gate.json}"

EPOCHS="${EPOCHS:-30}"
EVENTS_PER_CLASS="${EVENTS_PER_CLASS:-4}"
SAMPLES_PER_EVENT="${SAMPLES_PER_EVENT:-8}"
REFERENCE_CAP="${REFERENCE_CAP:-128}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${R1_OUTPUT_ROOT}"

preflight() {
  python -m py_compile \
    model/DamageFormer_S9_R1_CEAM.py \
    script/standard_ML/train_S9_R1_CEAM.py \
    script/standard_ML/s9_r1_ceam_common.py \
    script/standard_ML/build_S9_R1_reference.py \
    script/standard_ML/eval_S9_R1_CEAM.py
  python -m unittest tests/test_s9_r1_ceam.py -v
}

train() {
  test -s "${RAW_TRAIN}"
  python -u script/standard_ML/train_S9_R1_CEAM.py \
    --raw_train "${RAW_TRAIN}" \
    --output "${CEAM_CKPT}" \
    --history "${CEAM_HISTORY}" \
    --device cuda:0 \
    --embed_dim 128 \
    --hidden_dim 256 \
    --epochs "${EPOCHS}" \
    --learning_rate 2e-4 \
    --weight_decay 1e-3 \
    --events_per_class "${EVENTS_PER_CLASS}" \
    --samples_per_event "${SAMPLES_PER_EVENT}" \
    --dev_event_fraction 0.20 \
    --reference_cap_per_event_class "${REFERENCE_CAP}" \
    --eval_batch_size 2048 \
    --knn_topk 8 \
    --knn_beta 8.0 \
    --angular_scale 30.0 \
    --angular_margin 0.20 \
    --temperature 0.07 \
    --hard_negative_margin 0.20 \
    --lambda_supcon 0.20 \
    --lambda_hard 0.10 \
    --early_stop_patience 6 \
    --seed 3407
}

build_reference() {
  test -s "${RAW_TRAIN}"
  test -s "${CEAM_CKPT}"
  python -u script/standard_ML/build_S9_R1_reference.py \
    --raw_train "${RAW_TRAIN}" \
    --checkpoint "${CEAM_CKPT}" \
    --output "${REFERENCE_PATH}" \
    --report "${REFERENCE_REPORT}" \
    --device cuda:0 \
    --max_per_event_class "${REFERENCE_CAP}" \
    --batch_size 2048 \
    --seed 3407
}

validate() {
  test -s "${R0_PAIRED_VAL}"
  test -s "${RAW_VAL_PRED}"
  test -s "${CEAM_CKPT}"
  test -s "${REFERENCE_PATH}"
  python -u script/standard_ML/eval_S9_R1_CEAM.py \
    --paired_queries "${R0_PAIRED_VAL}" \
    --val_pred_queries "${RAW_VAL_PRED}" \
    --checkpoint "${CEAM_CKPT}" \
    --reference "${REFERENCE_PATH}" \
    --output "${VALIDATION_REPORT}" \
    --gate_output "${METRIC_GATE}" \
    --device cuda:0 \
    --batch_size 2048 \
    --knn_topk 8 \
    --knn_beta 8.0 \
    --max_neighbors_per_event 3 \
    --minimum_geometry_positive_ratio 0.70 \
    --minimum_selective_precision 0.50 \
    --maximum_gt_pred_macro_drop 0.02
}

case "${MODE}" in
  preflight) preflight ;;
  train) train ;;
  build_reference) build_reference ;;
  validate) validate ;;
  all)
    preflight
    train
    build_reference
    validate
    ;;
  *)
    echo "Unknown mode '${MODE}'" >&2
    echo "Expected preflight|train|build_reference|validate|all" >&2
    echo "No Test/prototype/fusion mode exists until metric_gate.json is accepted." >&2
    exit 2
    ;;
esac
