#!/usr/bin/env bash
set -euo pipefail

# Controlled E0--E4 experiment. Run from Linux training server:
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh preflight
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh stats
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh smoke e1
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh pilot e1
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh pilot e3a
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh audit e1
#   bash bda_benchmark/experiments/p5_ablation_validation_v1/run.sh compare c1_c0

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
BASE_CKPT="${BASE_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_BASE_SHA256="${EXPECTED_BASE_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/P5_ABLATION_VALIDATION_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/P5_ABLATION_VALIDATION_V1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

check_base() {
  test -s "${BASE_CKPT}" || { echo "Missing P3 checkpoint: ${BASE_CKPT}" >&2; exit 3; }
  local observed
  observed="$(sha256sum "${BASE_CKPT}" | awk '{print $1}')"
  [[ "${observed}" == "${EXPECTED_BASE_SHA256}" ]] || {
    echo "P3 checkpoint SHA256 mismatch: ${observed}" >&2
    exit 3
  }
}

experiment_contract() {
  case "$1" in
    e0) echo "p3 p3 E0_P3_frozen" ;;
    e1) echo "c0 flat E1_P5_C0_native_sar" ;;
    e2) echo "c1 flat E2_P5_C1_fixed_filter" ;;
    e3a) echo "c2_raw flat E3A_P5_C2Raw_native_sar_cross_attention" ;;
    e3) echo "c2 flat E3_P5_C2_local_cross_attention" ;;
    e4) echo "c2 hierarchical E4_P5_C2_hierarchical" ;;
    *) echo "target must be e0|e1|e2|e3a|e3|e4" >&2; exit 2 ;;
  esac
}

preflight() {
  python -m py_compile \
    experiments/p5_ablation_validation_v1/metrics.py \
    experiments/p5_ablation_validation_v1/audit_model.py \
    experiments/p5_ablation_validation_v1/inspect_status.py \
    experiments/p5_ablation_validation_v1/summarize_pair.py \
    script/standard_ML/train_S8_P5_dcmf.py \
    script/standard_ML/train_S8_P5_hierarchical.py
  python -m unittest discover -s experiments/p5_ablation_validation_v1/tests -v
  [[ ! -f "${BASE_CKPT}" ]] || check_base
  echo "Preflight passed. No model was trained and test was not accessed."
}

compute_stats() {
  mkdir -p "$(dirname "${SAR_STATS_JSON}")"
  if [[ -s "${SAR_STATS_JSON}" && "${FORCE_STATS:-0}" != "1" ]]; then
    echo "Reusing frozen train-only SAR statistics: ${SAR_STATS_JSON}"
    return
  fi
  python -u script/standard_ML/compute_S8_P4_sar_stats.py \
    --dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --output_json "${SAR_STATS_JSON}" \
    --sar_representation display \
    --seed "${SEED}"
}

train_one() {
  local target="$1" steps="$2" eval_every="$3" warmup="$4" tag="$5"
  read -r variant architecture name <<<"$(experiment_contract "${target}")"
  check_base
  test -s "${SAR_STATS_JSON}" || { echo "Run the stats stage first." >&2; exit 4; }
  local output_dir="${OUTPUT_ROOT}/${name}_${tag}_seed${SEED}"
  local trainer="script/standard_ML/train_S8_P5_dcmf.py"
  [[ "${architecture}" == "hierarchical" ]] && trainer="script/standard_ML/train_S8_P5_hierarchical.py"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/final.pth" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Completed checkpoint exists; skipped: ${output_dir}/final.pth"
    return
  fi
  local resume_args=()
  local resume_path="${RESUME_CKPT:-}"
  if [[ -z "${resume_path}" && -s "${output_dir}/latest_resume.pth" ]]; then
    resume_path="${output_dir}/latest_resume.pth"
  fi
  if [[ -n "${resume_path}" ]]; then
    test -s "${resume_path}" || { echo "Missing resume checkpoint: ${resume_path}" >&2; exit 7; }
    resume_args+=(--resume "${resume_path}" --skip_initial_eval)
    echo "Resuming from: ${resume_path}"
  fi
  local max_batches=0 mil_warmup=1000 guard=3000
  if (( steps <= 20 )); then max_batches=8; mil_warmup=5; guard=100; fi
  if (( steps == 5000 )); then guard=3000; fi
  if (( steps >= 8000 )); then guard=4000; fi
  python -u "${trainer}" \
    --variant "${variant}" \
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
    --optimizer_steps "${steps}" \
    --warmup_steps "${warmup}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --deterministic \
    --sar_learning_rate 1e-4 \
    --fusion_learning_rate 1e-4 \
    --core_learning_rate 1e-6 \
    --weight_decay 5e-3 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
    --mil_warmup_steps "${mil_warmup}" \
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
    --val_max_batches "${max_batches}" \
    --guard_start_step "${guard}" \
    --early_stop_drop_pp 1.0 \
    --early_stop_patience 2 \
    "${resume_args[@]}"
}

find_checkpoint() {
  local target="$1"
  if [[ "${target}" == "e0" ]]; then
    test -s "${BASE_CKPT}" || { echo "Missing checkpoint: ${BASE_CKPT}" >&2; exit 5; }
    echo "${BASE_CKPT}"
    return
  fi
  read -r _ _ name <<<"$(experiment_contract "${target}")"
  local requested="${OUTPUT_ROOT}/${name}_${TRAIN_TAG:-pilot5000}_seed${SEED}/best_val_miou.pth"
  test -s "${requested}" || { echo "Missing checkpoint: ${requested}" >&2; exit 5; }
  echo "${requested}"
}

audit_one() {
  local target="$1"
  read -r _ architecture name <<<"$(experiment_contract "${target}")"
  local checkpoint; checkpoint="$(find_checkpoint "${target}")"
  local output_dir="${AUDIT_ROOT}/${name}_${TRAIN_TAG:-pilot5000}_seed${SEED}"
  if [[ "${target}" == "e0" ]]; then
    output_dir="${AUDIT_ROOT}/${name}"
  fi
  mkdir -p "${output_dir}"
  local perturbations=(none shift_x4 shift_y4 shift_x8 shift_y8 pair_shuffle)
  [[ "${target}" == "e0" ]] && perturbations=(none)
  for perturbation in "${perturbations[@]}"; do
    local output_json="${output_dir}/${perturbation}.json"
    if [[ -s "${output_json}" && "${FORCE_AUDIT:-0}" != "1" ]]; then
      echo "Reusing completed audit: ${output_json}"
      continue
    fi
    local extra=()
    if [[ "${perturbation}" == "none" ]]; then
      extra+=(--gradient_probe)
      [[ "${target}" != "e0" ]] && extra+=(--frequency_png "${output_dir}/sar_frequency_views.png")
    fi
    python -u experiments/p5_ablation_validation_v1/audit_model.py \
      --architecture "${architecture}" \
      --checkpoint "${checkpoint}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_json "${SAR_STATS_JSON}" \
      --perturbation "${perturbation}" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --device cuda:0 \
      --output_json "${output_json}" \
      "${extra[@]}"
  done
}

compare_pair() {
  local key="$1" ref cand gate perturbation_name=shift_x4
  case "${key}" in
    c1_c0) ref=e1; cand=e2; gate=sar_filter ;;
    c2raw_c0) ref=e1; cand=e3a; gate=cross_attention ;;
    c2raw_c0_x8) ref=e1; cand=e3a; gate=cross_attention; perturbation_name=shift_x8 ;;
    c2raw_c0_y8) ref=e1; cand=e3a; gate=cross_attention; perturbation_name=shift_y8 ;;
    c2raw_c0_final) ref=e1; cand=e3a; gate=final ;;
    c2_c1) ref=e2; cand=e3; gate=cross_attention ;;
    c2_c1_x8) ref=e2; cand=e3; gate=cross_attention; perturbation_name=shift_x8 ;;
    c2_c1_y8) ref=e2; cand=e3; gate=cross_attention; perturbation_name=shift_y8 ;;
    c2_c2raw) ref=e3a; cand=e3; gate=filter_with_attention ;;
    c2_c0_final) ref=e1; cand=e3; gate=final ;;
    hier_flat) ref=e3; cand=e4; gate=hierarchical ;;
    final) ref=e0; cand=e4; gate=final ;;
    *) echo "comparison must be c1_c0|c2raw_c0|c2raw_c0_x8|c2raw_c0_y8|c2raw_c0_final|c2_c1|c2_c1_x8|c2_c1_y8|c2_c2raw|c2_c0_final|hier_flat|final" >&2; exit 6 ;;
  esac
  read -r _ _ ref_name <<<"$(experiment_contract "${ref}")"
  read -r _ _ cand_name <<<"$(experiment_contract "${cand}")"
  local ref_dir="${AUDIT_ROOT}/${ref_name}_${TRAIN_TAG:-pilot5000}_seed${SEED}"
  local cand_dir="${AUDIT_ROOT}/${cand_name}_${TRAIN_TAG:-pilot5000}_seed${SEED}"
  [[ "${ref}" == "e0" ]] && ref_dir="${AUDIT_ROOT}/${ref_name}"
  local extra=()
  if [[ "${gate}" == "cross_attention" ]]; then
    extra+=(
      --reference_perturbed "${ref_dir}/${perturbation_name}.json"
      --candidate_perturbed "${cand_dir}/${perturbation_name}.json"
      --reference_shuffled "${ref_dir}/pair_shuffle.json"
      --candidate_shuffled "${cand_dir}/pair_shuffle.json"
    )
  fi
  python -u experiments/p5_ablation_validation_v1/summarize_pair.py \
    --reference "${ref_dir}/none.json" \
    --candidate "${cand_dir}/none.json" \
    --gate "${gate}" \
    --output_json "${AUDIT_ROOT}/${key}_admission.json" \
    --output_md "${AUDIT_ROOT}/${key}_admission.md" \
    "${extra[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  stats) compute_stats ;;
  smoke) train_one "${TARGET}" 20 10 5 smoke20 ;;
  pilot) train_one "${TARGET}" 5000 500 1000 pilot5000 ;;
  full) train_one "${TARGET}" 8000 500 1000 full8000 ;;
  audit) audit_one "${TARGET}" ;;
  compare) compare_pair "${TARGET}" ;;
  status)
    echo "Active training processes:"
    pgrep -af 'train_S8_P5_(dcmf|hierarchical).py' || true
    echo
    echo "GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
    python -u experiments/p5_ablation_validation_v1/inspect_status.py \
      --output_root "${OUTPUT_ROOT}" \
      --log_root "${LOG_ROOT:-}"
    ;;
  *) echo "mode must be preflight|stats|smoke|pilot|full|audit|compare|status" >&2; exit 2 ;;
esac
