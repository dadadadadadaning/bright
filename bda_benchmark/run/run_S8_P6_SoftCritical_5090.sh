#!/usr/bin/env bash
set -euo pipefail

# Isolated S8-P6 experiment.  Validation only; no test-set argument exists.
#
#   bash run/run_S8_P6_SoftCritical_5090.sh preflight
#   bash run/run_S8_P6_SoftCritical_5090.sh stats
#   bash run/run_S8_P6_SoftCritical_5090.sh exact_guard
#   bash run/run_S8_P6_SoftCritical_5090.sh smoke
#   bash run/run_S8_P6_SoftCritical_5090.sh pilot4000
#   bash run/run_S8_P6_SoftCritical_5090.sh gate

MODE="${1:-pilot4000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
VARIANT=a4
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P6_SOFTCRITICAL}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OCCURRENCE_TEMPERATURE="${OCCURRENCE_TEMPERATURE:-0.5}"
CORE_LR="${CORE_LR:-1e-6}"
INPUT_LR="${INPUT_LR:-1e-4}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export OCCURRENCE_TEMPERATURE

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
    model/DamageFormer_S8_MICB_DSMIL.py \
    model/DamageFormer_S8_P6_SoftCritical.py \
    script/standard_ML/train_S8_P6_softcritical.py \
    script/standard_ML/summarize_S8_P6_softcritical.py
  python -m unittest tests/test_s8_micb_dsmil.py -v
  python -m unittest tests/test_s8_p4_native_sar.py -v
  python -m unittest tests/test_s8_p6_softcritical.py -v
  if [[ -f "${BASE_CKPT}" ]]; then
    check_base
  fi
  echo "S8-P6 SoftCritical preflight passed."
}

compute_stats() {
  mkdir -p "$(dirname "${SAR_STATS_JSON}")"
  if [[ -s "${SAR_STATS_JSON}" && "${FORCE_STATS:-0}" != "1" ]]; then
    echo "Frozen train-only SAR statistics exist: ${SAR_STATS_JSON}"
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
  local input_only_steps="$3"
  local tag="$4"
  check_base
  test -s "${SAR_STATS_JSON}" || {
    echo "Missing train-only SAR statistics: ${SAR_STATS_JSON}" >&2
    echo "Run: bash run/run_S8_P6_SoftCritical_5090.sh stats" >&2
    exit 4
  }

  local exp_name="${EXP_NAME:-S8_P6_A4_SoftCritical_${tag}_seed${SEED}}"
  local output_dir="${OUTPUT_ROOT}/${exp_name}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/final.pth" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Final checkpoint exists; skipped: ${output_dir}/final.pth"
    return 0
  fi

  local resume_args=()
  if [[ -s "${output_dir}/latest_resume.pth" ]]; then
    resume_args+=(--resume "${output_dir}/latest_resume.pth" --skip_initial_eval)
    echo "Auto resume: ${output_dir}/latest_resume.pth"
  fi
  local val_max_batches=0
  local mil_warmup=1000
  if (( optimizer_steps > 0 && optimizer_steps <= 20 )); then
    val_max_batches=8
    mil_warmup=5
  fi

  echo "================================================================================"
  echo "Experiment    : ${exp_name}"
  echo "Model         : P3 + A4 native SAR + Soft-Critical MICB"
  echo "Temperature   : ${OCCURRENCE_TEMPERATURE}"
  echo "CrossAttention: DISABLED"
  echo "MIL residual  : 0 (training auxiliary only)"
  echo "Steps         : ${optimizer_steps}; A4-only=${input_only_steps}, then A4 locked"
  echo "Selection     : validation valid_mIoU; initial safe state included"
  echo "Test access   : forbidden"
  echo "Output        : ${output_dir}"
  echo "================================================================================"

  python -u script/standard_ML/train_S8_P6_softcritical.py \
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
    --num_workers 8 \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --core_learning_rate "${CORE_LR}" \
    --input_learning_rate "${INPUT_LR}" \
    --weight_decay 5e-3 \
    --input_only_steps "${input_only_steps}" \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
    --mil_warmup_steps "${mil_warmup}" \
    --lambda_mil 0.25 \
    --lambda_exist_aux 0.25 \
    --lambda_severity_aux 0.25 \
    --lambda_intact_guard 0.05 \
    --lambda_coverage 0.0 \
    --kd_temperature 2.0 \
    --lambda_kd_damage 1.0 \
    --lambda_kd_location 0.25 \
    --lambda_kd_feature 0.1 \
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 500 \
    --val_max_batches "${val_max_batches}" \
    --expected_a0_valid_miou 60.5019 \
    --a0_tolerance 0.0001 \
    --early_stop_drop_pp 0.3 \
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
  exact_guard)
    train_run 0 1 0 exact_guard
    ;;
  smoke)
    train_run 20 10 5 smoke20
    ;;
  pilot4000)
    train_run 4000 500 1000 pilot4000
    ;;
  gate)
    baseline_summary="${BASELINE_SUMMARY:-${OUTPUT_ROOT}/S8_P6_A4_SoftCritical_exact_guard_seed${SEED}/training_summary.json}"
    candidate_summary="${CANDIDATE_SUMMARY:-${OUTPUT_ROOT}/S8_P6_A4_SoftCritical_pilot4000_seed${SEED}/training_summary.json}"
    test -s "${baseline_summary}"
    test -s "${candidate_summary}"
    python -u script/standard_ML/summarize_S8_P6_softcritical.py \
      --baseline "${baseline_summary}" \
      --candidate "${candidate_summary}" \
      --output "${OUTPUT_ROOT}/S8_P6_A4_SoftCritical_pilot4000_seed${SEED}/advancement_gate.json"
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|exact_guard|smoke|pilot4000|gate" >&2
    exit 2
    ;;
esac
