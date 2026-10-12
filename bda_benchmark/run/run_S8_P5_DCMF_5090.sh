#!/usr/bin/env bash
set -euo pipefail

# Independent one-GPU launcher for S8-P5 DCMF.
#
#   bash run/run_S8_P5_DCMF_5090.sh preflight
#   bash run/run_S8_P5_DCMF_5090.sh stats
#   VARIANT=c3 bash run/run_S8_P5_DCMF_5090.sh smoke
#   VARIANT=c3 bash run/run_S8_P5_DCMF_5090.sh pilot

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
VARIANT="${VARIANT:-c3}"
VARIANT="${VARIANT,,}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P5_DCMF}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_P5_DCMF}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
EXPECTED_A0_VALID_MIOU="${EXPECTED_A0_VALID_MIOU:-60.5019}"

case "${VARIANT}" in
  c0|c1|c2|c3) ;;
  *)
    echo "VARIANT must be c0|c1|c2|c3, got '${VARIANT}'" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

check_base() {
  test -s "${BASE_CKPT}"
  local observed
  observed="$(sha256sum "${BASE_CKPT}" | awk '{print $1}')"
  if [[ "${observed}" != "${EXPECTED_BASE_SHA256}" ]]; then
    echo "P3 SHA256 mismatch: ${observed}" >&2
    echo "Expected: ${EXPECTED_BASE_SHA256}" >&2
    exit 3
  fi
}

preflight() {
  python -m py_compile \
    model/DamageFormer_S8_P5_DCMF.py \
    script/standard_ML/train_S8_P5_dcmf.py
  python -m unittest tests/test_s8_micb_dsmil.py -v
  python -m unittest tests/test_s8_p5_dcmf.py -v
  if [[ -f "${BASE_CKPT}" ]]; then
    check_base
  fi
  echo "S8-P5 DCMF preflight passed."
}

compute_stats() {
  mkdir -p "$(dirname "${SAR_STATS_JSON}")"
  if [[ -s "${SAR_STATS_JSON}" && "${FORCE_STATS:-0}" != "1" ]]; then
    echo "Frozen train-only SAR statistics already exist: ${SAR_STATS_JSON}"
    return 0
  fi
  python -u script/standard_ML/compute_S8_P4_sar_stats.py \
    --dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --output_json "${SAR_STATS_JSON}" \
    --sar_representation display \
    --seed "${SEED}"
}

train_run() {
  local optimizer_steps="$1"
  local eval_every="$2"
  local warmup_steps="$3"
  local guard_start_step="$4"
  local mode_tag="$5"
  local mil_warmup_steps=1000
  local val_max_batches=0
  if (( optimizer_steps <= 20 )); then
    mil_warmup_steps=5
    val_max_batches=8
  fi

  check_base
  test -s "${SAR_STATS_JSON}" || {
    echo "Missing train-only SAR stats: ${SAR_STATS_JSON}" >&2
    echo "Run: bash run/run_S8_P5_DCMF_5090.sh stats" >&2
    exit 4
  }

  local variant_upper="${VARIANT^^}"
  local default_name="S8_P5_${variant_upper}_${mode_tag}_seed${SEED}"
  local exp_name="${EXP_NAME:-${default_name}}"
  local output_dir="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${output_dir}" "${LOG_ROOT}"

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
    test -s "${resume_path}"
    resume_args+=(--resume "${resume_path}" --skip_initial_eval)
    echo "Auto resume: ${resume_path}"
  fi

  echo "================================================================================"
  echo "Experiment : ${exp_name}"
  echo "Variant    : ${VARIANT}"
  echo "Base       : ${BASE_CKPT}"
  echo "Stats      : ${SAR_STATS_JSON}"
  echo "Output     : ${output_dir}"
  echo "Steps      : ${optimizer_steps}; warmup=${warmup_steps}; guard=${guard_start_step}"
  echo "================================================================================"

  python -u script/standard_ML/train_S8_P5_dcmf.py \
    --variant "${VARIANT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --base_ckpt "${BASE_CKPT}" \
    --output_dir "${output_dir}" \
    --crop_size 640 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --accumulation_steps 4 \
    --optimizer_steps "${optimizer_steps}" \
    --warmup_steps "${warmup_steps}" \
    --num_workers 8 \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --sar_learning_rate 1e-4 \
    --fusion_learning_rate 1e-4 \
    --core_learning_rate 1e-6 \
    --weight_decay 5e-3 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
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
    --lambda_decur_common 0.05 \
    --lambda_decur_unique 0.02 \
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 1000 \
    --val_max_batches "${val_max_batches}" \
    --expected_a0_valid_miou "${EXPECTED_A0_VALID_MIOU}" \
    --guard_start_step "${guard_start_step}" \
    --early_stop_drop_pp 1.0 \
    --early_stop_patience 2 \
    "${resume_args[@]}"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  stats)
    compute_stats
    ;;
  smoke)
    train_run 20 10 5 100  smoke20
    ;;
  pilot)
    train_run 5000 500 1000 3000 pilot5000
    ;;
  full)
    train_run 8000 500 1000 4000 full8000
    ;;
  paper50k)
    train_run 50000 1000 1000 5000 paper50000
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|smoke|pilot|full|paper50k" >&2
    exit 2
    ;;
esac
