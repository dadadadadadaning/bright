#!/usr/bin/env bash
set -euo pipefail

# P3 + filtered-SAR local Cross-Attention, added as a zero-gated residual.
# This experiment reuses the completed E3 checkpoint and never overwrites P3/P5.

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
E3_CKPT="${E3_CKPT:-/root/data/BRIGHT-checkpoints/P5_ABLATION_VALIDATION_V1/E3_P5_C2_local_cross_attention_pilot5000_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/P3_P5_SAFE_RESIDUAL_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/P3_P5_SAFE_RESIDUAL_V1}"
REFERENCE_AUDIT="${REFERENCE_AUDIT:-/root/shared-nvme/BRIGHT-eval-results/P5_ABLATION_VALIDATION_V1/E0_P3_frozen/none.json}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

check_inputs() {
  test -s "${P3_CKPT}" || { echo "Missing P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  test -s "${E3_CKPT}" || { echo "Missing completed E3 checkpoint: ${E3_CKPT}" >&2; exit 4; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing frozen SAR statistics: ${SAR_STATS_JSON}" >&2; exit 5; }
  test -s "${TRAIN_LIST}" || { echo "Missing train list: ${TRAIN_LIST}" >&2; exit 6; }
  test -s "${VAL_LIST}" || { echo "Missing validation list: ${VAL_LIST}" >&2; exit 7; }
}

preflight() {
  python -m py_compile \
    experiments/p3_p5_safe_residual_v1/audit.py \
    script/standard_ML/train_S8_P5_dcmf.py \
    model/DamageFormer_S8_P5_C2_Recovery.py
  check_inputs
  python - "${E3_CKPT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
variant = str(payload.get("variant", "")).lower()
if variant != "c2":
    raise SystemExit(f"E3 checkpoint must contain variant=c2, got {variant!r}")
print(f"E3 checkpoint verified: step={payload.get('step')} variant={variant}")
PY
  echo "Preflight passed. P3 and E3 are read-only inputs; official test was not accessed."
}

train_one() {
  local tag="$1" steps="$2" eval_every="$3" warmup_steps="$4" max_batches="$5"
  check_inputs
  local output_dir="${OUTPUT_ROOT}/P3_E3_zero_gated_residual_${tag}_seed${SEED}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/final.pth" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Completed stage exists; skipped: ${output_dir}"
    return
  fi
  local resume_args=()
  if [[ -s "${output_dir}/latest_resume.pth" ]]; then
    resume_args+=(--resume "${output_dir}/latest_resume.pth" --skip_initial_eval)
  fi
  python -u script/standard_ML/train_S8_P5_dcmf.py \
    --variant c2 \
    --recovery_variant r1 \
    --c2_checkpoint "${E3_CKPT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --base_ckpt "${P3_CKPT}" \
    --output_dir "${output_dir}" \
    --crop_size 640 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --accumulation_steps 4 \
    --optimizer_steps "${steps}" \
    --warmup_steps "${warmup_steps}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --device cuda:0 \
    --amp_dtype bf16 \
    --seed "${SEED}" \
    --deterministic \
    --sar_learning_rate 1e-4 \
    --fusion_learning_rate 1e-4 \
    --core_learning_rate 1e-6 \
    --recovery_learning_rate 1e-3 \
    --weight_decay 5e-3 \
    --min_instance_area 32 \
    --max_bags_per_class_per_batch 16 \
    --mil_warmup_steps "${warmup_steps}" \
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
    --log_every 20 \
    --eval_every "${eval_every}" \
    --save_every 1000 \
    --val_max_batches "${max_batches}" \
    --guard_start_step 3000 \
    --early_stop_drop_pp 0.30 \
    --early_stop_patience 2 \
    "${resume_args[@]}"
}

checkpoint_for_audit() {
  local tag="${TRAIN_TAG:-probe5000}"
  local directory="${OUTPUT_ROOT}/P3_E3_zero_gated_residual_${tag}_seed${SEED}"
  local name="${AUDIT_WEIGHTS:-best_val_miou}.pth"
  [[ "${AUDIT_WEIGHTS:-best_val_miou}" == "latest" ]] && name="latest_resume.pth"
  test -s "${directory}/${name}" || { echo "Missing candidate checkpoint: ${directory}/${name}" >&2; exit 8; }
  echo "${directory}/${name}"
}

audit_one() {
  check_inputs
  local checkpoint
  checkpoint="$(checkpoint_for_audit)"
  local tag="${TRAIN_TAG:-probe5000}_${AUDIT_WEIGHTS:-best_val_miou}"
  local output_dir="${AUDIT_ROOT}/${tag}_seed${SEED}"
  mkdir -p "${output_dir}"
  for perturbation in none shift_x4 shift_y4 shift_x8 shift_y8 pair_shuffle; do
    local output_json="${output_dir}/${perturbation}.json"
    if [[ -s "${output_json}" && "${FORCE_AUDIT:-0}" != "1" ]]; then
      echo "Reusing completed audit: ${output_json}"
      continue
    fi
    python -u experiments/p3_p5_safe_residual_v1/audit.py \
      --checkpoint "${checkpoint}" \
      --base_ckpt "${P3_CKPT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_json "${SAR_STATS_JSON}" \
      --perturbation "${perturbation}" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --device cuda:0 \
      --output_json "${output_json}"
  done
}

compare_one() {
  local tag="${TRAIN_TAG:-probe5000}_${AUDIT_WEIGHTS:-best_val_miou}"
  local candidate_dir="${AUDIT_ROOT}/${tag}_seed${SEED}"
  test -s "${REFERENCE_AUDIT}" || {
    echo "Missing P3 reference audit: ${REFERENCE_AUDIT}" >&2
    echo "Create it once with: bash experiments/p5_ablation_validation_v1/run.sh audit e0" >&2
    exit 9
  }
  test -s "${candidate_dir}/none.json" || { echo "Run candidate audit first." >&2; exit 10; }
  python -u experiments/p5_ablation_validation_v1/summarize_pair.py \
    --reference "${REFERENCE_AUDIT}" \
    --candidate "${candidate_dir}/none.json" \
    --gate final \
    --output_json "${AUDIT_ROOT}/p3_e3_residual_admission.json" \
    --output_md "${AUDIT_ROOT}/p3_e3_residual_admission.md"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) train_one smoke20 20 10 5 8 ;;
  probe) train_one probe5000 5000 500 1000 0 ;;
  audit) audit_one ;;
  compare) compare_one ;;
  status)
    pgrep -af 'train_S8_P5_dcmf.py' || true
    python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*/training_summary.json")):
    data = json.loads(path.read_text())
    print(path.parent.name)
    print("  completed_step:", data.get("completed_step"))
    print("  best_step:", data.get("best_step"))
    print("  best_valid_miou:", data.get("best_valid_miou"))
    print("  recovery_gate:", data.get("recovery_gate"))
PY
    ;;
  *) echo "mode must be preflight|smoke|probe|audit|compare|status" >&2; exit 2 ;;
esac
