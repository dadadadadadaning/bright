#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED}"
VARIANT="${VARIANT:-sr1}"
SEED="${SEED:-3407}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export P3_CKPT
export H6_A2_CKPT

preflight() {
  python -m py_compile \
    model/DamageFormer_P3_SRAC_Controlled.py \
    script/standard_ML/p3_srac_controlled_runtime.py \
    script/standard_ML/train_P3_SRAC_Controlled.py \
    script/standard_ML/check_P3_SRAC_guard.py \
    script/standard_ML/summarize_P3_SRAC.py \
    script/standard_ML/summarize_P3_SRAC_partition.py \
    script/standard_ML/summarize_P3_SRAC_affinity.py
  python -m unittest tests.test_p3_srac_controlled -v
  python -c 'from dataset.make_data_loader_heterogeneous_rgbsar import VALID_HETEROGENEOUS_RGBSAR_VARIANTS as v; assert {"sr1f", "sr1p", "sr1pa", "sr1pb", "sr1pai", "sr1pib", "sr1paa"}.issubset(v), v'
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 1; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing SAR statistics: ${SAR_STATS_JSON}" >&2; exit 1; }
  if [[ "${VARIANT}" == "sr1paa" ]]; then
    test -s "${H6_A2_CKPT}" || { echo "Missing H6-A2 checkpoint: ${H6_A2_CKPT}" >&2; exit 1; }
  fi
  echo "P3-SRAC controlled preflight passed."
}

maybe_preflight() {
  if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    preflight
  fi
}

run_train() {
  local steps="$1"
  local output_dir="$2"
  local extra=()
  local selection_metric="event_worst_official_style_4class_miou"
  if [[ "${VARIANT}" == "sr1paa" ]]; then
    test -s "${H6_A2_CKPT}" || {
      echo "SR1PAA requires H6_A2_CKPT with a frozen native-SAR encoder" >&2
      exit 1
    }
  fi
  if [[ "${steps}" != "0" && -s "${output_dir}/latest_resume.pth" \
        && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "Refusing to overwrite an existing training run: ${output_dir}" >&2
    echo "Set a new EXP_NAME (recommended), or ALLOW_EXISTING_OUTPUT=1 deliberately." >&2
    exit 1
  fi
  if [[ "${steps}" == "0" ]]; then
    extra+=(--validation_only)
  fi
  if [[ "${VARIANT}" == "sr1p" || "${VARIANT}" == "sr1pb" \
        || "${VARIANT}" == "sr1pib" ]]; then
    extra+=(--select_initial_checkpoint)
  fi
  if [[ "${VARIANT}" == "sr1pa" || "${VARIANT}" == "sr1pai" ]]; then
    selection_metric="srac_partition_quality"
  elif [[ "${VARIANT}" == "sr1paa" ]]; then
    selection_metric="srac_affinity_balanced_accuracy"
  fi
  if [[ "${VARIANT}" == "sr1pb" ]]; then
    test -s "${SR1PA_CKPT:-}" || {
      echo "SR1PB requires SR1PA_CKPT from completed partition pretraining" >&2
      exit 1
    }
  fi
  if [[ "${VARIANT}" == "sr1pib" ]]; then
    test -s "${SR1PAI_CKPT:-}" || {
      echo "SR1PIB requires SR1PAI_CKPT from completed iterative partition pretraining" >&2
      exit 1
    }
  fi
  if [[ -n "${RESUME_CKPT:-}" ]]; then
    test -s "${RESUME_CKPT}" || {
      echo "Missing resume/evaluation checkpoint: ${RESUME_CKPT}" >&2
      exit 1
    }
    extra+=(--resume "${RESUME_CKPT}")
  fi
  python -u script/standard_ML/train_P3_SRAC_Controlled.py \
    --variant "${VARIANT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_path "${SAR_STATS_JSON}" \
    --output_dir "${output_dir}" \
    --device cuda:0 \
    --crop_size "${CROP_SIZE:-640}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --eval_batch_size 1 \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" \
    --optimizer_steps "${steps}" \
    --eval_every "${EVAL_EVERY:-500}" \
    --save_every "${SAVE_EVERY:-1000}" \
    --log_every 20 \
    --num_workers "${NUM_WORKERS:-8}" \
    --rgb_learning_rate 0.0 \
    --sar_learning_rate 0.0 \
    --fusion_learning_rate "${LEARNING_RATE:-0.0001}" \
    --damage_ce_weight 1.0 \
    --damage_lovasz_weight 0.75 \
    --sar_aux_weight 0.0 \
    --representation_weight "${REGION_LOSS_WEIGHT:-0.25}" \
    --selection_metric "${selection_metric}" \
    --amp bf16 \
    --seed "${SEED}" \
    --error_on_nonfinite_grad \
    "${extra[@]}"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  guard)
    maybe_preflight
    VARIANT=sr0 run_train 0 "${OUTPUT_ROOT}/P3_SR0_guard_seed${SEED}"
    guard_args=(
      --p3-checkpoint "${P3_CKPT}"
      --sr0-metrics "${OUTPUT_ROOT}/P3_SR0_guard_seed${SEED}/validation_metrics.json"
    )
    if [[ -n "${P3_REFERENCE_JSON:-}" ]]; then
      guard_args+=(--reference-metrics "${P3_REFERENCE_JSON}")
    fi
    python script/standard_ML/check_P3_SRAC_guard.py "${guard_args[@]}"
    ;;
  smoke)
    maybe_preflight
    run_train 20 "${OUTPUT_ROOT}/${EXP_NAME:-P3_${VARIANT^^}_smoke20_seed${SEED}}"
    ;;
  screen)
    maybe_preflight
    run_train "${OPTIMIZER_STEPS:-3000}" \
      "${OUTPUT_ROOT}/${EXP_NAME:-P3_${VARIANT^^}_screen${OPTIMIZER_STEPS:-3000}_seed${SEED}}"
    ;;
  eval_val)
    maybe_preflight
    : "${SRAC_CKPT:?Set SRAC_CKPT to the selected SR1P checkpoint}"
    eval_name="${EVAL_NAME:-P3_${VARIANT^^}_independent_val_seed${SEED}}"
    RESUME_CKPT="${SRAC_CKPT}" run_train 0 "${OUTPUT_ROOT}/${eval_name}"
    ;;
  *)
    echo "Unknown mode ${MODE}; expected preflight|guard|smoke|screen|eval_val" >&2
    exit 2
    ;;
esac
