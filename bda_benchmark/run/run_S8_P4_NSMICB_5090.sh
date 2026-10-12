#!/usr/bin/env bash
set -euo pipefail

# One-GPU launcher for S8-P4 NS-MICB A0/A1/A2/A3/A4.
#
# Examples:
#   bash run/run_S8_P4_NSMICB_5090.sh preflight
#   bash run/run_S8_P4_NSMICB_5090.sh stats
#   VARIANT=a0 bash run/run_S8_P4_NSMICB_5090.sh guard
#   bash run/run_S8_P4_NSMICB_5090.sh exact_guard
#   VARIANT=a2 bash run/run_S8_P4_NSMICB_5090.sh smoke
#   VARIANT=a2 bash run/run_S8_P4_NSMICB_5090.sh pilot

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
VARIANT="${VARIANT:-a0}"
VARIANT="${VARIANT,,}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${OUTPUT_ROOT}/sar_train_stats.json}"
SAR_REPRESENTATION="${SAR_REPRESENTATION:-display}"
EXPECTED_A0_VALID_MIOU="${EXPECTED_A0_VALID_MIOU:-60.5019}"
A0_TOLERANCE="${A0_TOLERANCE:-0.10}"
CORE_LR="${CORE_LR:-1e-6}"
INPUT_LR="${INPUT_LR:-1e-4}"

case "${VARIANT}" in
  a0|a1|a2|a3|a4) ;;
  *)
    echo "VARIANT must be a0|a1|a2|a3|a4, got '${VARIANT}'" >&2
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
    model/DamageFormer_S8_P4_NativeSAR_MICB.py \
    dataset/make_data_loader_p4_native_sar.py \
    script/standard_ML/compute_S8_P4_sar_stats.py \
    script/standard_ML/train_S8_P4_native_sar_micb.py \
    script/standard_ML/eval_S8_P4_native_sar_micb.py \
    script/standard_ML/summarize_S8_P4_ablation.py
  python -m unittest tests/test_s8_micb_dsmil.py -v
  python -m unittest tests/test_s8_p4_native_sar.py -v
  if [[ -f "${BASE_CKPT}" ]]; then
    check_base
  fi
  echo "S8-P4 preflight passed."
}

compute_stats() {
  mkdir -p "$(dirname "${SAR_STATS_JSON}")"
  if [[ -s "${SAR_STATS_JSON}" && "${FORCE_STATS:-0}" != "1" ]]; then
    echo "Frozen SAR statistics already exist: ${SAR_STATS_JSON}"
    return 0
  fi
  python -u script/standard_ML/compute_S8_P4_sar_stats.py \
    --dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --output_json "${SAR_STATS_JSON}" \
    --sar_representation "${SAR_REPRESENTATION}" \
    --seed "${SEED}"
}

train_run() {
  local optimizer_steps="$1"
  local eval_every="$2"
  local input_only_steps="$3"
  local mode_tag="$4"
  local mil_warmup_steps="${MIL_WARMUP_STEPS_OVERRIDE:-1000}"
  local val_max_batches=0
  if (( optimizer_steps > 0 && optimizer_steps <= 20 )); then
    mil_warmup_steps=5
    val_max_batches=8
  fi
  check_base
  if [[ "${VARIANT}" != "a0" ]]; then
    test -s "${SAR_STATS_JSON}" || {
      echo "Missing SAR stats: ${SAR_STATS_JSON}" >&2
      echo "Run: bash run/run_S8_P4_NSMICB_5090.sh stats" >&2
      exit 4
    }
  fi

  local variant_upper="${VARIANT^^}"
  local default_name="S8_P4_${variant_upper}_${mode_tag}_seed${SEED}"
  local exp_name="${EXP_NAME:-${default_name}}"
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
    test -s "${resume_path}"
    resume_args+=(--resume "${resume_path}" --skip_initial_eval)
    if [[ "${EXPECTED_RESUME_STEP:-}" != "" ]]; then
      resume_args+=(--expected_resume_step "${EXPECTED_RESUME_STEP}")
    fi
    if [[ "${EXPECTED_RESUME_VALID_MIOU_PERCENT:-}" != "" ]]; then
      resume_args+=(
        --expected_resume_valid_miou_percent "${EXPECTED_RESUME_VALID_MIOU_PERCENT}"
        --resume_metric_tolerance_pp "${RESUME_METRIC_TOLERANCE_PP:-0.02}"
      )
    fi
    if [[ "${ADMIT_RESUME_CHECKPOINT:-0}" == "1" ]]; then
      resume_args+=(--admit_resume_checkpoint)
    fi
    echo "Auto resume: ${resume_path}"
  fi

  echo "================================================================================"
  echo "Experiment : ${exp_name}"
  echo "Variant    : ${VARIANT} (R1/R4/reliability all disabled)"
  echo "Base       : ${BASE_CKPT}"
  echo "Stats      : ${SAR_STATS_JSON}"
  echo "Output     : ${output_dir}"
  echo "Steps      : ${optimizer_steps}; input-only=${input_only_steps}"
  echo "================================================================================"

  python -u script/standard_ML/train_S8_P4_native_sar_micb.py \
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
    --mil_warmup_steps "${mil_warmup_steps}" \
    --lambda_mil "${LAMBDA_MIL:-0.25}" \
    --lambda_exist_aux "${LAMBDA_EXIST_AUX:-0.25}" \
    --lambda_severity_aux "${LAMBDA_SEVERITY_AUX:-0.25}" \
    --lambda_intact_guard "${LAMBDA_INTACT_GUARD:-0.05}" \
    --lambda_coverage "${LAMBDA_COVERAGE:-0.0}" \
    --kd_temperature 2.0 \
    --lambda_kd_damage 1.0 \
    --lambda_kd_location 0.25 \
    --lambda_kd_feature 0.1 \
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 1000 \
    --val_max_batches "${val_max_batches}" \
    --expected_a0_valid_miou "${EXPECTED_A0_VALID_MIOU}" \
    --a0_tolerance "${A0_TOLERANCE}" \
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
  guard)
    train_run 0 1 0 guard
    ;;
  exact_guard)
    # A4 is the corrected function-preserving one-channel control.  It must
    # reproduce P3 before any optimisation; do not silently turn this into a
    # training run.
    VARIANT=a4
    A0_TOLERANCE="${A4_TOLERANCE:-0.0001}"
    train_run 0 1 0 exact_guard
    ;;
  smoke)
    train_run 20 10 5 smoke20
    ;;
  pilot)
    train_run 3000 500 1000 pilot3000
    ;;
  joint_no_mil)
    # Causal control: begin at the validated A4 step-1000 input-adaptation
    # state, unlock exactly fusion_layer_3 + clf_2, and disable every
    # MICB/DSMIL term.  The target step is 1500, i.e. exactly 500 new updates.
    VARIANT=a4
    RESUME_CKPT="${A4_STEP1000_CKPT:-${RESUME_CKPT:-${OUTPUT_ROOT}/S8_P4_A4_pilot3000_seed${SEED}/checkpoint_step_1000.pth}}"
    if [[ -z "${RESUME_CKPT}" || ! -s "${RESUME_CKPT}" ]]; then
      echo "Missing A4 step-1000 checkpoint. Set A4_STEP1000_CKPT=/path/to/step1000.pth" >&2
      exit 5
    fi
    EXPECTED_RESUME_STEP=1000
    EXPECTED_RESUME_VALID_MIOU_PERCENT="${EXPECTED_A4_STEP1000_VALID_MIOU:-60.5751}"
    ADMIT_RESUME_CHECKPOINT=1
    MIL_WARMUP_STEPS_OVERRIDE=999999
    LAMBDA_MIL=0.0
    LAMBDA_EXIST_AUX=0.0
    LAMBDA_SEVERITY_AUX=0.0
    LAMBDA_INTACT_GUARD=0.0
    LAMBDA_COVERAGE=0.0
    EXP_NAME="${EXP_NAME:-S8_P4_A4_joint_noMIL_step1000_to1500_seed${SEED}}"
    train_run 1500 250 1000 joint_noMIL500
    ;;
  full)
    train_run 50000 500 1000 full50000
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|guard|exact_guard|smoke|pilot|joint_no_mil|full" >&2
    exit 2
    ;;
esac
