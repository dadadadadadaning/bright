#!/usr/bin/env bash
set -euo pipefail

# P6-C1 resumes the validated P6/A4 step-1000 state and tests only whether a
# low-authority Soft-Critical occurrence signal can improve weak Damaged recall.
# It never accesses the test split and never enables Memory retrieval.

MODE="${1:-screen1000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
P6_STEP1000="${P6_STEP1000:-/root/shared-nvme/BRIGHT-checkpoints/S8_P6_SOFTCRITICAL/S8_P6_A4_SoftCritical_pilot4000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P6C1_OCCURRENCE}"
OCCURRENCE_TEMPERATURE="${OCCURRENCE_TEMPERATURE:-0.5}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export OCCURRENCE_TEMPERATURE

preflight() {
  python -m py_compile \
    model/DamageFormer_S8_P6_SoftCritical.py \
    script/standard_ML/train_S8_P4_native_sar_micb.py \
    script/standard_ML/train_S8_P6C1_occurrence.py
  python -m unittest tests/test_s8_p6_softcritical.py -v
  echo "S8-P6C1 occurrence-only preflight passed."
}

run_screen() {
  local target_step="$1"
  local eval_every="$2"
  local tag="$3"
  test -s "${BASE_CKPT}"
  test -s "${SAR_STATS_JSON}"
  test -s "${P6_STEP1000}" || {
    echo "Missing validated P6 step-1000 checkpoint: ${P6_STEP1000}" >&2
    exit 4
  }
  local exp_name="${EXP_NAME:-S8_P6C1_occurrence_${tag}_seed${SEED}}"
  local output_dir="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${output_dir}"

  python -u script/standard_ML/train_S8_P6C1_occurrence.py \
    --variant a4 \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --base_ckpt "${BASE_CKPT}" \
    --resume "${P6_STEP1000}" \
    --expected_resume_step 1000 \
    --expected_resume_valid_miou_percent 60.5738 \
    --resume_metric_tolerance_pp 0.03 \
    --admit_resume_checkpoint \
    --output_dir "${output_dir}" \
    --crop_size 640 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --accumulation_steps 4 \
    --optimizer_steps "${target_step}" \
    --num_workers 8 \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --core_learning_rate 5e-7 \
    --input_learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --input_only_steps 1000 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
    --mil_warmup_steps 1000 \
    --mil_ramp_steps 500 \
    --lambda_mil 0.0 \
    --lambda_exist_aux 0.05 \
    --lambda_severity_aux 0.0 \
    --lambda_intact_guard 0.02 \
    --lambda_coverage 0.0 \
    --kd_temperature 2.0 \
    --lambda_kd_damage 1.0 \
    --lambda_kd_location 0.25 \
    --lambda_kd_feature 0.1 \
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 250 \
    --val_max_batches 0 \
    --expected_a0_valid_miou 60.5738 \
    --a0_tolerance 0.03 \
    --early_stop_drop_pp 0.10 \
    --early_stop_patience 2
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) run_screen 1020 20 smoke20 ;;
  screen1000) run_screen 2000 250 screen1000 ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|screen1000" >&2
    exit 2
    ;;
esac
