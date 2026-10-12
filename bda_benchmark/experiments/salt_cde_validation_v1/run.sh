#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
TARGET="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SALT_CDE_VALIDATION_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/SALT_CDE_VALIDATION_V1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

name_of() {
  case "$1" in
    d0) echo "D0_original_damageformer" ;;
    d0n) echo "D0N_native_sar_colocated" ;;
    d1) echo "D1_local_transport" ;;
    d2) echo "D2_stable_ring_transport" ;;
    d3) echo "D3_counterfactual_stable_ring" ;;
    d2r) echo "D2R_safe_residual_ring_transport" ;;
    d3v2) echo "D3V2_explicit_pair_verified_transport" ;;
    *) echo "target must be d0|d0n|d1|d2|d3|d2r|d3v2" >&2; exit 2 ;;
  esac
}

preflight() {
  python -m py_compile \
    bda_benchmark/model/DamageFormer_SALT_CDE.py \
    bda_benchmark/script/standard_ML/train_SALT_CDE.py \
    bda_benchmark/experiments/salt_cde_validation_v1/audit_model.py \
    bda_benchmark/experiments/salt_cde_validation_v1/compare.py \
    bda_benchmark/experiments/salt_cde_validation_v1/summarize_curve.py
  python -m unittest discover \
    -s bda_benchmark/experiments/salt_cde_validation_v1/tests -v
  echo "Preflight passed; no training and no test access."
}

train_one() {
  local target="$1" steps="$2" tag="$3" previous_tag="$4"
  local name output_dir
  name="$(name_of "${target}")"
  output_dir="${OUTPUT_ROOT}/${name}_${tag}_seed${SEED}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/training_summary.json" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Completed experiment exists; skipped: ${output_dir}"
    return
  fi
  local resume_args=()
  local variant_args=()
  [[ "${target}" == "d3" ]] && variant_args+=(--base_aux_weight 0.25)
  [[ "${target}" == "d3v2" ]] && variant_args+=(--base_aux_weight 0.25 --counterfactual_weight 0.10)
  local resume="${RESUME_CKPT:-}"
  if [[ -z "${resume}" && -s "${output_dir}/latest_resume.pth" ]]; then
    resume="${output_dir}/latest_resume.pth"
  fi
  if [[ -z "${resume}" && -n "${previous_tag}" && "${FRESH_STAGE:-0}" != "1" ]]; then
    local candidate_tag previous
    for candidate_tag in ${previous_tag//,/ }; do
      previous="${OUTPUT_ROOT}/${name}_${candidate_tag}_seed${SEED}/latest_resume.pth"
      if [[ -s "${previous}" ]]; then
        resume="${previous}"
        break
      fi
    done
  fi
  if [[ -n "${resume}" ]]; then
    local inherited_best
    inherited_best="$(dirname "${resume}")/best_val_miou.pth"
    if [[ -s "${inherited_best}" && ! -s "${output_dir}/best_val_miou.pth" ]]; then
      cp "${inherited_best}" "${output_dir}/best_val_miou.pth"
      echo "Inherited earlier best weights: ${inherited_best}"
    fi
    resume_args+=(--resume "${resume}" --skip_initial_eval)
    echo "Continuing optimizer state from: ${resume}"
  fi
  local val_max=0 eval_every=1000 save_every=1000
  (( steps <= 20 )) && val_max=8 && eval_every=10 && save_every=10
  python -u bda_benchmark/script/standard_ML/train_SALT_CDE.py \
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
    --loc_ce_weight 1.0 \
    --loc_lovasz_weight 1.0 \
    --damage_ce_weight 1.0 \
    --damage_lovasz_weight 1.0 \
    --seed "${SEED}" \
    "${variant_args[@]}" \
    "${resume_args[@]}"
}

checkpoint_of() {
  local target="$1" name path
  name="$(name_of "${target}")"
  path="${OUTPUT_ROOT}/${name}_${TRAIN_TAG:-pilot8000}_seed${SEED}/best_val_miou.pth"
  test -s "${path}" || { echo "Missing checkpoint: ${path}" >&2; exit 4; }
  echo "${path}"
}

audit_one() {
  local target="$1" name checkpoint output_dir
  name="$(name_of "${target}")"
  checkpoint="$(checkpoint_of "${target}")"
  output_dir="${AUDIT_ROOT}/${name}_${TRAIN_TAG:-pilot8000}_seed${SEED}"
  mkdir -p "${output_dir}"
  local modes=(none shift_x4 shift_y4 shift_x8 shift_y8 shift_x16 shift_y16 pair_shuffle zero)
  for mode in "${modes[@]}"; do
    if [[ -s "${output_dir}/${mode}.json" && "${FORCE_AUDIT:-0}" != "1" ]]; then
      echo "Reusing ${output_dir}/${mode}.json"
      continue
    fi
    python -u bda_benchmark/experiments/salt_cde_validation_v1/audit_model.py \
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

compare_one() {
  local pair="$1" reference candidate perturbation
  case "${pair}" in
    d0n_d0) reference=d0; candidate=d0n; perturbation=none ;;
    d1_d0n) reference=d0n; candidate=d1; perturbation=shift_x8 ;;
    d2_d1) reference=d1; candidate=d2; perturbation=shift_x8 ;;
    d3_d2) reference=d2; candidate=d3; perturbation=pair_shuffle ;;
    d2r_d1) reference=d1; candidate=d2r; perturbation=shift_x8 ;;
    d3v2_d2r) reference=d2r; candidate=d3v2; perturbation=pair_shuffle ;;
    *) echo "pair must be d0n_d0|d1_d0n|d2_d1|d3_d2|d2r_d1|d3v2_d2r" >&2; exit 5 ;;
  esac
  local rn cn rd cd extra=()
  rn="$(name_of "${reference}")"; cn="$(name_of "${candidate}")"
  rd="${AUDIT_ROOT}/${rn}_${TRAIN_TAG:-pilot8000}_seed${SEED}"
  cd="${AUDIT_ROOT}/${cn}_${TRAIN_TAG:-pilot8000}_seed${SEED}"
  if [[ "${perturbation}" != "none" ]]; then
    extra+=(--reference_perturbed "${rd}/${perturbation}.json" --candidate_perturbed "${cd}/${perturbation}.json")
  fi
  python -u bda_benchmark/experiments/salt_cde_validation_v1/compare.py \
    --reference "${rd}/none.json" \
    --candidate "${cd}/none.json" \
    --output_json "${AUDIT_ROOT}/${pair}_${TRAIN_TAG:-pilot8000}_admission.json" \
    "${extra[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) train_one "${TARGET}" 20 smoke20 "" ;;
  probe) train_one "${TARGET}" 2000 probe2000 "" ;;
  pilot) train_one "${TARGET}" 8000 pilot8000 probe2000 ;;
  short) train_one "${TARGET}" 20000 short20000 pilot8000 ;;
  long) train_one "${TARGET}" 35000 long35000 short20000,pilot8000 ;;
  final) train_one "${TARGET}" 50000 final50000 long35000,short20000,pilot8000 ;;
  audit) audit_one "${TARGET}" ;;
  curve)
    curve_name="$(name_of "${TARGET}")"
    curve_dir="${AUDIT_ROOT}/${curve_name}_${TRAIN_TAG:-pilot8000}_seed${SEED}"
    python -u bda_benchmark/experiments/salt_cde_validation_v1/summarize_curve.py \
      --audit_dir "${curve_dir}" \
      --output_json "${curve_dir}/perturbation_curve.json"
    ;;
  compare) compare_one "${TARGET}" ;;
  status)
    pgrep -af 'train_SALT_CDE.py' || true
    find "${OUTPUT_ROOT}" -maxdepth 2 -name training_summary.json -print 2>/dev/null || true
    ;;
  *) echo "mode must be preflight|smoke|probe|pilot|short|long|final|audit|curve|compare|status" >&2; exit 2 ;;
esac
