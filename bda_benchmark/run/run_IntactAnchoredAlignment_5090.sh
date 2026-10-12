#!/usr/bin/env bash
set -euo pipefail

# Isolated AL0/AL0-N/AL1--AL3 protocol.  AL0-N additionally reads a frozen P3
# checkpoint only to define target-free building support for scene statistics.
# Outputs stay below ALIGNED_DAMAGE_EVIDENCE; no mode exposes a test-list input.

MODE="${1:-sequence}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/ALIGNED_DAMAGE_EVIDENCE}"
EVAL_ROOT="${EVAL_ROOT:-/root/shared-nvme/BRIGHT-eval-results/ALIGNED_DAMAGE_EVIDENCE}"
AL1_DIR="${AL1_DIR:-${OUTPUT_ROOT}/AL1_intact_alignment_screen1500_seed${SEED}}"
AL1_CKPT="${AL1_CKPT:-${AL1_DIR}/best_intact_alignment.pth}"
AL1_SMOKE_DIR="${AL1_SMOKE_DIR:-${OUTPUT_ROOT}/AL1_intact_alignment_smoke20_seed${SEED}}"
AL0_JSON="${AL0_JSON:-${EVAL_ROOT}/AL0_untrained_descriptor_val_seed${SEED}.json}"
AL0N_JSON="${AL0N_JSON:-${EVAL_ROOT}/AL0N_scene_relative_unaligned_residual_val_seed${SEED}.json}"
AL2_JSON="${AL2_JSON:-${EVAL_ROOT}/AL2_frozen_aligned_residual_val_seed${SEED}.json}"
AL3_DIR="${AL3_DIR:-${OUTPUT_ROOT}/AL3_occurrence_screen1500_seed${SEED}}"
AL3_SMOKE_DIR="${AL3_SMOKE_DIR:-${OUTPUT_ROOT}/AL3_occurrence_smoke20_seed${SEED}}"

common_args=(
  --dataset_path "${DATASET_ROOT}"
  --train_data_list_path "${TRAIN_LIST}"
  --val_data_list_path "${VAL_LIST}"
  --sar_stats_path "${SAR_STATS_JSON}"
  --h6_a2_checkpoint "${H6_A2_CKPT}"
  --device cuda:0
  --crop_size "${CROP_SIZE:-640}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}"
  --num_workers "${NUM_WORKERS:-8}"
  --descriptor_channels "${DESCRIPTOR_CHANNELS:-64}"
  --radius "${ALIGN_RADIUS:-2}"
  --temperature "${ALIGN_TEMPERATURE:-0.10}"
  --seed "${SEED}"
)

preflight() {
  bash -n run/run_IntactAnchoredAlignment_5090.sh
  python -m py_compile \
    model/IntactAnchoredAlignment.py \
    script/standard_ML/al_alignment_runtime.py \
    script/standard_ML/train_IntactAnchoredAlignment.py
  python -m unittest -v tests.test_intact_anchored_alignment
  echo "Intact-anchored AL0/AL0-N/AL1--AL3 preflight passed."
}

check_inputs() {
  for required in "${TRAIN_LIST}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${H6_A2_CKPT}"; do
    test -s "${required}" || { echo "Missing AL input: ${required}" >&2; exit 3; }
  done
  mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"
}

run_al0() {
  check_inputs
  python -u script/standard_ML/train_IntactAnchoredAlignment.py \
    --stage al0 --output_json "${AL0_JSON}" "${common_args[@]}"
}

run_al0n() {
  check_inputs
  test -s "${P3_CKPT}" || { echo "Missing frozen P3 checkpoint: ${P3_CKPT}" >&2; exit 4; }
  python -u script/standard_ML/train_IntactAnchoredAlignment.py \
    --stage al0n --p3_checkpoint "${P3_CKPT}" --output_json "${AL0N_JSON}" \
    --building_probability_threshold "${BUILDING_PROBABILITY_THRESHOLD:-0.50}" \
    --minimum_scene_support "${MINIMUM_SCENE_SUPPORT:-16}" \
    --minimum_occurrence_event_gain "${MINIMUM_OCCURRENCE_EVENT_GAIN:-0.02}" \
    "${common_args[@]}"
}

run_al1() {
  local steps="$1"
  local output_dir="$2"
  check_inputs
  mkdir -p "${output_dir}"
  python -u script/standard_ML/train_IntactAnchoredAlignment.py \
    --stage al1 --output_dir "${output_dir}" \
    --optimizer_steps "${steps}" --eval_every "${EVAL_EVERY:-500}" \
    --log_every "${LOG_EVERY:-20}" --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --learning_rate "${ALIGN_LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --smoothness_weight "${SMOOTHNESS_WEIGHT:-0.05}" \
    --magnitude_weight "${MAGNITUDE_WEIGHT:-0.01}" \
    --variance_weight "${VARIANCE_WEIGHT:-0.10}" \
    --intact_focus_probability "${INTACT_FOCUS_PROBABILITY:-0.75}" \
    "${common_args[@]}"
}

run_al2() {
  check_inputs
  test -s "${AL1_CKPT}" || { echo "Missing admitted AL1 checkpoint: ${AL1_CKPT}" >&2; exit 4; }
  python -u script/standard_ML/train_IntactAnchoredAlignment.py \
    --stage al2 --alignment_checkpoint "${AL1_CKPT}" \
    --output_json "${AL2_JSON}" "${common_args[@]}"
}

al2_passed() {
  python - "${AL2_JSON}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("passed") is True else 1)
PY
}

run_al3() {
  local steps="$1"
  local output_dir="$2"
  check_inputs
  test -s "${AL1_CKPT}" || { echo "Missing AL1 checkpoint: ${AL1_CKPT}" >&2; exit 4; }
  test -s "${AL2_JSON}" || { echo "Missing AL2 gate: ${AL2_JSON}" >&2; exit 5; }
  al2_passed || {
    echo "AL2 gate failed. AL3 is intentionally blocked; no classifier was trained." >&2
    exit 6
  }
  mkdir -p "${output_dir}"
  python -u script/standard_ML/train_IntactAnchoredAlignment.py \
    --stage al3 --alignment_checkpoint "${AL1_CKPT}" \
    --al2_gate_json "${AL2_JSON}" --output_dir "${output_dir}" \
    --optimizer_steps "${steps}" --eval_every "${EVAL_EVERY:-500}" \
    --log_every "${LOG_EVERY:-20}" --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --learning_rate "${OCCURRENCE_LR:-1e-4}" \
    --affected_weight "${AFFECTED_WEIGHT:-2.0}" \
    "${common_args[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  al0) run_al0 ;;
  al0n) run_al0n ;;
  al1_smoke) run_al1 20 "${AL1_SMOKE_DIR}" ;;
  al1_screen) run_al1 "${OPTIMIZER_STEPS:-1500}" "${AL1_DIR}" ;;
  al2) run_al2 ;;
  al3_smoke) run_al3 20 "${AL3_SMOKE_DIR}" ;;
  al3_screen) run_al3 "${OPTIMIZER_STEPS:-1500}" "${AL3_DIR}" ;;
  sequence)
    preflight
    run_al0
    run_al1 20 "${AL1_SMOKE_DIR}"
    run_al1 "${OPTIMIZER_STEPS:-1500}" "${AL1_DIR}"
    run_al2
    if al2_passed; then
      echo "AL2 passed. AL3 remains a separate explicit command:"
      echo "  bash run/run_IntactAnchoredAlignment_5090.sh al3_smoke"
    else
      echo "Decision: STOP_BEFORE_AL3_ALIGNMENT_EVIDENCE_INSUFFICIENT"
    fi
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|al0|al0n|al1_smoke|al1_screen|al2|al3_smoke|al3_screen|sequence" >&2
    exit 2
    ;;
esac
