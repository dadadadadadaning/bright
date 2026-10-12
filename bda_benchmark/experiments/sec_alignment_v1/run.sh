#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
TARGET="${2:-}"
REFERENCE_TARGET="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SEC_ALIGNMENT_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/SEC_ALIGNMENT_V1}"
SALT_OUTPUT_ROOT="${SALT_OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SALT_CDE_VALIDATION_V1}"
SALT_AUDIT_ROOT="${SALT_AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/SALT_CDE_VALIDATION_V1}"
D1_NAME="D1_local_transport"
D1_TAG="${D1_TAG:-pilot8000}"
D1_CKPT="${D1_CKPT:-${SALT_OUTPUT_ROOT}/${D1_NAME}_${D1_TAG}_seed${SEED}/best_val_miou.pth}"
D1_AUDIT_DIR="${D1_AUDIT_DIR:-${AUDIT_ROOT}/REFERENCE_${D1_NAME}_${D1_TAG}_seed${SEED}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

name_of() {
  case "$1" in
    s1) echo "S1_shift_augmentation_control" ;;
    s2) echo "S2_relative_offset_equivariance" ;;
    s3) echo "S3_offset_prediction_consistency" ;;
    s2r) echo "S2R_safe_query_key_alignment" ;;
    s2r0) echo "S2R0_full_image_safe_alignment" ;;
    s2rp) echo "S2RP_predicted_building_safe_alignment" ;;
    s2rr) echo "S2RR_predicted_building_ring_safe_alignment" ;;
    s2rcf) echo "S2RCF_building_guided_coarse_to_fine_alignment" ;;
    *) echo "target must be s1|s2|s3|s2r|s2r0|s2rp|s2rr|s2rcf" >&2; exit 2 ;;
  esac
}

preflight() {
  test -s "${D1_CKPT}" || { echo "Missing completed D1 checkpoint: ${D1_CKPT}" >&2; exit 3; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing frozen train-only SAR statistics: ${SAR_STATS_JSON}" >&2; exit 3; }
  test -s "${TRAIN_LIST}" && test -s "${VAL_LIST}" || { echo "Missing train/validation list" >&2; exit 3; }
  python -m py_compile \
    bda_benchmark/experiments/sec_alignment_v1/objectives.py \
    bda_benchmark/experiments/sec_alignment_v1/model.py \
    bda_benchmark/experiments/sec_alignment_v1/train.py \
    bda_benchmark/experiments/sec_alignment_v1/audit.py \
    bda_benchmark/experiments/sec_alignment_v1/compare.py
  python -m unittest discover \
    -s bda_benchmark/experiments/sec_alignment_v1/tests -v
  echo "SEC preflight passed; completed D1 will be reused, not retrained."
}

train_one() {
  local target="$1" steps="$2" tag="$3" previous_tag="$4"
  local name output_dir resume=""
  name="$(name_of "${target}")"
  output_dir="${OUTPUT_ROOT}/${name}_${tag}_seed${SEED}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/training_summary.json" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Completed SEC stage exists; skipped: ${output_dir}"
    return
  fi
  local resume_args=()
  if [[ -s "${output_dir}/latest_resume.pth" ]]; then
    resume="${output_dir}/latest_resume.pth"
  elif [[ -n "${previous_tag}" && "${FRESH_STAGE:-0}" != "1" ]]; then
    local previous="${OUTPUT_ROOT}/${name}_${previous_tag}_seed${SEED}/latest_resume.pth"
    [[ -s "${previous}" ]] && resume="${previous}"
  fi
  if [[ -n "${resume}" ]]; then
    local inherited_best
    inherited_best="$(dirname "${resume}")/best_val_miou.pth"
    if [[ -s "${inherited_best}" && ! -s "${output_dir}/best_val_miou.pth" ]]; then
      cp "${inherited_best}" "${output_dir}/best_val_miou.pth"
      echo "Inherited earlier best SEC weights: ${inherited_best}"
    fi
    resume_args+=(--resume "${resume}" --skip_initial_eval)
    echo "Continuing the same SEC variant from: ${resume}"
  else
    test -s "${D1_CKPT}" || { echo "Missing completed D1 checkpoint: ${D1_CKPT}" >&2; exit 3; }
    resume_args+=(--init_checkpoint "${D1_CKPT}")
    echo "Starting ${target} from the same frozen D1 checkpoint."
  fi
  local val_max=0 eval_every=1000 save_every=1000
  (( steps <= 20 )) && val_max=8 && eval_every=10 && save_every=10
  python -u bda_benchmark/experiments/sec_alignment_v1/train.py \
    --variant "${target}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_path "${SAR_STATS_JSON}" \
    --output_dir "${output_dir}" \
    --crop_size 640 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --accumulation_steps 4 \
    --optimizer_steps "${steps}" \
    --eval_every "${eval_every}" \
    --save_every "${save_every}" \
    --log_every 20 \
    --num_workers "${NUM_WORKERS:-8}" \
    --val_max_batches "${val_max}" \
    --device cuda:0 \
    --amp bf16 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --shift_amounts 4 8 \
    --offset_delta_weight 0.05 \
    --consistency_weight 0.10 \
    --objective_warmup_steps 500 \
    --seed "${SEED}" \
    "${resume_args[@]}"
}

candidate_dir() {
  local name
  name="$(name_of "$1")"
  echo "${OUTPUT_ROOT}/${name}_${TRAIN_TAG:-pilot8000}_seed${SEED}"
}

audit_weights_file() {
  case "${AUDIT_WEIGHTS:-best}" in
    best) echo "best_val_miou.pth" ;;
    latest) echo "latest_resume.pth" ;;
    *) echo "AUDIT_WEIGHTS must be best|latest" >&2; exit 4 ;;
  esac
}

audit_suffix() {
  [[ "${AUDIT_WEIGHTS:-best}" == "latest" ]] && echo "_latest" || echo ""
}

audit_one() {
  local target="$1" candidate checkpoint output_dir
  candidate="$(candidate_dir "${target}")"
  checkpoint="${candidate}/$(audit_weights_file)"
  test -s "${checkpoint}" || { echo "Missing SEC checkpoint: ${checkpoint}" >&2; exit 4; }
  output_dir="${AUDIT_ROOT}/$(basename "${candidate}")$(audit_suffix)"
  mkdir -p "${output_dir}"
  for mode in none shift_x4 shift_y4 shift_x8 shift_y8 shift_x16 shift_y16; do
    if [[ -s "${output_dir}/${mode}.json" && "${FORCE_AUDIT:-0}" != "1" ]]; then
      echo "Reusing ${output_dir}/${mode}.json"
      continue
    fi
    python -u bda_benchmark/experiments/sec_alignment_v1/audit.py \
      --checkpoint "${checkpoint}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --perturbation "${mode}" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --device cuda:0 \
      --output_json "${output_dir}/${mode}.json"
  done
}

audit_reference() {
  test -s "${D1_CKPT}" || { echo "Missing completed D1 checkpoint: ${D1_CKPT}" >&2; exit 3; }
  mkdir -p "${D1_AUDIT_DIR}"
  for mode in none shift_x4 shift_y4 shift_x8 shift_y8 shift_x16 shift_y16; do
    if [[ -s "${D1_AUDIT_DIR}/${mode}.json" && "${FORCE_AUDIT:-0}" != "1" ]]; then
      echo "Reusing ${D1_AUDIT_DIR}/${mode}.json"
      continue
    fi
    python -u bda_benchmark/experiments/sec_alignment_v1/audit.py \
      --checkpoint "${D1_CKPT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --perturbation "${mode}" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --device cuda:0 \
      --output_json "${D1_AUDIT_DIR}/${mode}.json"
  done
}

compare_one() {
  local target="$1" candidate audit_dir
  candidate="$(candidate_dir "${target}")"
  audit_dir="${AUDIT_ROOT}/$(basename "${candidate}")$(audit_suffix)"
  for mode in none shift_x8 shift_y8; do
    test -s "${D1_AUDIT_DIR}/${mode}.json" || { echo "Missing existing D1 audit ${D1_AUDIT_DIR}/${mode}.json" >&2; exit 5; }
    test -s "${audit_dir}/${mode}.json" || { echo "Run SEC audit first; missing ${audit_dir}/${mode}.json" >&2; exit 5; }
  done
  python -u bda_benchmark/experiments/sec_alignment_v1/compare.py \
    --reference_clean "${D1_AUDIT_DIR}/none.json" \
    --candidate_clean "${audit_dir}/none.json" \
    --reference_x8 "${D1_AUDIT_DIR}/shift_x8.json" \
    --reference_y8 "${D1_AUDIT_DIR}/shift_y8.json" \
    --candidate_x8 "${audit_dir}/shift_x8.json" \
    --candidate_y8 "${audit_dir}/shift_y8.json" \
    --output_json "${AUDIT_ROOT}/${target}_vs_d1_${TRAIN_TAG:-pilot8000}$(audit_suffix)_admission.json"
}

compare_pair() {
  local target="$1" reference_target="$2"
  [[ -n "${reference_target}" ]] || {
    echo "compare_pair requires: candidate reference" >&2; exit 2;
  }
  local candidate reference candidate_audit reference_audit
  candidate="$(candidate_dir "${target}")"
  reference="$(candidate_dir "${reference_target}")"
  candidate_audit="${AUDIT_ROOT}/$(basename "${candidate}")$(audit_suffix)"
  reference_audit="${AUDIT_ROOT}/$(basename "${reference}")$(audit_suffix)"
  for mode in none shift_x8 shift_y8; do
    test -s "${reference_audit}/${mode}.json" || {
      echo "Missing reference audit ${reference_audit}/${mode}.json" >&2; exit 5;
    }
    test -s "${candidate_audit}/${mode}.json" || {
      echo "Missing candidate audit ${candidate_audit}/${mode}.json" >&2; exit 5;
    }
  done
  python -u bda_benchmark/experiments/sec_alignment_v1/compare.py \
    --reference_clean "${reference_audit}/none.json" \
    --candidate_clean "${candidate_audit}/none.json" \
    --reference_x8 "${reference_audit}/shift_x8.json" \
    --reference_y8 "${reference_audit}/shift_y8.json" \
    --candidate_x8 "${candidate_audit}/shift_x8.json" \
    --candidate_y8 "${candidate_audit}/shift_y8.json" \
    --output_json "${AUDIT_ROOT}/${target}_vs_${reference_target}_${TRAIN_TAG:-pilot8000}$(audit_suffix)_admission.json"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) train_one "${TARGET}" 20 smoke20 "" ;;
  probe) train_one "${TARGET}" 5000 probe5000 "" ;;
  pilot) train_one "${TARGET}" 8000 pilot8000 probe5000 ;;
  short) train_one "${TARGET}" 20000 short20000 pilot8000 ;;
  final) train_one "${TARGET}" 50000 final50000 short20000 ;;
  audit) audit_one "${TARGET}" ;;
  audit_reference) audit_reference ;;
  compare) compare_one "${TARGET}" ;;
  compare_pair) compare_pair "${TARGET}" "${REFERENCE_TARGET}" ;;
  status)
    pgrep -af 'sec_alignment_v1/train.py' || true
    find "${OUTPUT_ROOT}" -maxdepth 2 -name training_summary.json -print 2>/dev/null || true
    ;;
  *) echo "mode must be preflight|smoke|probe|pilot|short|final|audit_reference|audit|compare|compare_pair|status" >&2; exit 2 ;;
esac
