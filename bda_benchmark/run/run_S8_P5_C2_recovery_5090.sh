#!/usr/bin/env bash
set -euo pipefail

# One-GPU launcher for the controlled C2 recovery experiments.
#
#   bash run/run_S8_P5_C2_recovery_5090.sh preflight
#   RECOVERY_VARIANT=r1 bash run/run_S8_P5_C2_recovery_5090.sh guard
#   RECOVERY_VARIANT=r2 bash run/run_S8_P5_C2_recovery_5090.sh smoke
#   RECOVERY_VARIANT=r2 bash run/run_S8_P5_C2_recovery_5090.sh pilot

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
RECOVERY_VARIANT="${RECOVERY_VARIANT:-r1}"
RECOVERY_VARIANT="${RECOVERY_VARIANT,,}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ACCUMULATION_STEPS))
MAX_BAGS_PER_CLASS_PER_BATCH="${MAX_BAGS_PER_CLASS_PER_BATCH:-$((4 * TRAIN_BATCH_SIZE))}"
if (( TRAIN_BATCH_SIZE < 1 || EVAL_BATCH_SIZE < 1 || ACCUMULATION_STEPS < 1 )); then
  echo "Batch sizes and accumulation steps must be positive." >&2
  exit 2
fi
if (( NUM_WORKERS < 0 )); then
  echo "NUM_WORKERS must be non-negative." >&2
  exit 2
fi
if (( EFFECTIVE_BATCH_SIZE != 16 )) && [[ "${ALLOW_EBS_CHANGE:-0}" != "1" ]]; then
  echo "Effective batch size must remain 16 for controlled experiments; " \
       "got ${TRAIN_BATCH_SIZE}x${ACCUMULATION_STEPS}=${EFFECTIVE_BATCH_SIZE}." >&2
  exit 2
fi
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
C2_CKPT="${C2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P5_DCMF/S8_P5_C2_pilot5000_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P5_C2_Recovery}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_P5_C2_Recovery}"
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
EXPECTED_BASE_VALID_MIOU="${EXPECTED_BASE_VALID_MIOU:-60.5019}"

case "${RECOVERY_VARIANT}" in
  r0|r1|r2|r3) ;;
  *)
    echo "RECOVERY_VARIANT must be r0|r1|r2|r3" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

require_nonempty_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "Required ${label} is missing or empty: ${path}" >&2
    return 1
  fi
}

preflight() {
  python -m py_compile \
    model/DamageFormer_S8_P5_C2_Recovery.py \
    script/standard_ML/train_S8_P5_dcmf.py
  python -m unittest tests/test_s8_p5_c2_recovery.py -v
  require_nonempty_file "${BASE_CKPT}" "P3 base checkpoint"
  require_nonempty_file "${C2_CKPT}" "C2 source checkpoint"
  require_nonempty_file "${SAR_STATS_JSON}" "SAR statistics JSON"
  local observed
  observed="$(sha256sum "${BASE_CKPT}" | awk '{print $1}')"
  if [[ "${observed}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "P3 SHA256 mismatch: ${observed}" >&2
    exit 3
  fi
  echo "S8-P5 C2 recovery preflight passed."
}

train_run() {
  local optimizer_steps="$1"
  local eval_every="$2"
  local warmup_steps="$3"
  local guard_start_step="$4"
  local patience="$5"
  local tag="$6"
  local val_max_batches=0
  local mil_warmup_steps=1000
  local mode_args=()
  if [[ "${tag}" == "smoke20" ]]; then
    val_max_batches=8
    mil_warmup_steps=5
  elif [[ "${tag}" == "guard" ]]; then
    mode_args+=(--validation_only_no_checkpoint)
  fi
  if [[ "${RECOVERY_VARIANT}" == "r0" ]]; then
    warmup_steps=0
  fi

  require_nonempty_file "${BASE_CKPT}" "P3 base checkpoint"
  require_nonempty_file "${C2_CKPT}" "C2 source checkpoint"
  require_nonempty_file "${SAR_STATS_JSON}" "SAR statistics JSON"
  mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

  local upper="${RECOVERY_VARIANT^^}"
  local exp_name="${EXP_NAME:-S8_P5_C2_${upper}_${tag}_seed${SEED}}"
  local output_dir="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${output_dir}"

  if [[ -s "${output_dir}/final.pth" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Final checkpoint exists; skipped: ${output_dir}/final.pth"
    return 0
  fi

  local resume_path="${RESUME_CKPT:-}"
  if [[ -z "${resume_path}" && -s "${output_dir}/latest_resume.pth" ]]; then
    resume_path="${output_dir}/latest_resume.pth"
  fi
  local resume_args=()
  if [[ -n "${resume_path}" ]]; then
    resume_args+=(--resume "${resume_path}" --skip_initial_eval)
    echo "Auto resume: ${resume_path}"
  fi

  echo "================================================================================"
  echo "Experiment       : ${exp_name}"
  echo "Recovery variant : ${RECOVERY_VARIANT}"
  echo "P3 anchor        : ${BASE_CKPT}"
  echo "C2 source        : ${C2_CKPT}"
  echo "Output           : ${output_dir}"
  echo "Steps/warmup     : ${optimizer_steps}/${warmup_steps}"
  echo "Batch tr/eval/acc: ${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}/${ACCUMULATION_STEPS}"
  echo "Effective batch  : ${EFFECTIVE_BATCH_SIZE}; workers=${NUM_WORKERS}"
  echo "Max bags/class   : ${MAX_BAGS_PER_CLASS_PER_BATCH} per micro-batch"
  echo "Guard/patience   : ${guard_start_step}/${patience}"
  echo "================================================================================"

  python -u script/standard_ML/train_S8_P5_dcmf.py \
    --variant c2 \
    --recovery_variant "${RECOVERY_VARIANT}" \
    --c2_checkpoint "${C2_CKPT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --base_ckpt "${BASE_CKPT}" \
    --output_dir "${output_dir}" \
    --crop_size 640 \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --accumulation_steps "${ACCUMULATION_STEPS}" \
    --optimizer_steps "${optimizer_steps}" \
    --warmup_steps "${warmup_steps}" \
    --num_workers "${NUM_WORKERS}" \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --sar_learning_rate 1e-5 \
    --fusion_learning_rate 1e-5 \
    --core_learning_rate 5e-7 \
    --recovery_learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch "${MAX_BAGS_PER_CLASS_PER_BATCH}" \
    --mil_warmup_steps "${mil_warmup_steps}" \
    --lambda_mil 0.25 \
    --lambda_exist_aux 0.25 \
    --lambda_severity_aux 0.25 \
    --lambda_intact_guard 0.05 \
    --kd_temperature 2.0 \
    --lambda_kd_damage 1.0 \
    --lambda_kd_location 0.25 \
    --lambda_kd_fused 0.10 \
    --lambda_kd_sar_warmup 1.0 \
    --lambda_kd_sar_late 0.25 \
    --lambda_decur_common 0.0 \
    --lambda_decur_unique 0.0 \
    --error_flow_margin 0.30 \
    --lambda_error_flow 0.10 \
    --supcon_temperature 0.10 \
    --lambda_supcon 0.05 \
    --recovery_loss_warmup_steps 500 \
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 1000 \
    --val_max_batches "${val_max_batches}" \
    --expected_a0_valid_miou "${EXPECTED_BASE_VALID_MIOU}" \
    --anchor_init_tolerance_pp 0.05 \
    --guard_start_step "${guard_start_step}" \
    --early_stop_drop_pp 1.0 \
    --early_stop_patience "${patience}" \
    "${mode_args[@]}" \
    "${resume_args[@]}"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  guard)
    train_run 0 1 500 5000 0 guard
    ;;
  smoke)
    train_run 20 10 5 100 0 smoke20
    ;;
  pilot)
    train_run 6000 500 500 5000 3 pilot6000
    ;;
  full)
    train_run 8000 500 500 6000 3 full8000
    ;;
  *)
    echo "Choose preflight|guard|smoke|pilot|full" >&2
    exit 2
    ;;
esac
