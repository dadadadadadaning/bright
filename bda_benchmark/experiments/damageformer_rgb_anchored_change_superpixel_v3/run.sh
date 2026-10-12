#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-p5_anchored_change}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${VARIANT}" in
  p2_rgb_structure|p5_anchored_change|p6_shuffled_sar|p7_no_confidence|p8_no_damage_boundary) ;;
  *) echo "Unknown V3 variant: ${VARIANT}" >&2; exit 2 ;;
esac

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3}"
SEED="${SEED:-3407}"
DEVICE="${DEVICE:-cuda:0}"
BASE_OUT="${OUTPUT_ROOT}/${VARIANT}_seed${SEED}"
STAGE0_OUT="${BASE_OUT}/stage0_structure"
STAGE1_OUT="${BASE_OUT}/stage1_change"

common_args() {
  printf '%s\n' \
    --variant "${VARIANT}" --dataset_path "${DATASET_PATH}" \
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --a1_ckpt "${A1_CKPT}" \
    --batch_size "${BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" --num_workers "${NUM_WORKERS:-8}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" --device "${DEVICE}" \
    --amp_dtype "${AMP_DTYPE:-bf16}" --seed "${SEED}" \
    --target_region_area "${TARGET_REGION_AREA:-16}" --min_region_area "${MIN_REGION_AREA:-4}" \
    --max_regions_per_building "${MAX_REGIONS_PER_BUILDING:-16}" \
    --building_threshold "${BUILDING_THRESHOLD:-0.50}" --min_building_area "${MIN_BUILDING_AREA:-8}" \
    --change_alpha_max "${CHANGE_ALPHA_MAX:-0.30}" --peak_temperature "${PEAK_TEMPERATURE:-0.20}" \
    --bag_aggregator "${BAG_AGGREGATOR:-logmeanexp}" --bag_temperature "${BAG_TEMPERATURE:-0.50}" \
    --bag_topk "${BAG_TOPK:-3}" --bag_label_purity "${BAG_LABEL_PURITY:-0.70}"
}

preflight() {
  python -m py_compile experiments/damageformer_rgb_anchored_change_superpixel_v3/*.py
  python -m unittest experiments.damageformer_rgb_anchored_change_superpixel_v3.test_contract -v
  test -s "${A1_CKPT}" || { echo "Missing A1 checkpoint: ${A1_CKPT}" >&2; exit 3; }
  test -s "${TRAIN_LIST}" || { echo "Missing train list: ${TRAIN_LIST}" >&2; exit 3; }
  test -s "${VAL_LIST}" || { echo "Missing val list: ${VAL_LIST}" >&2; exit 3; }
  [[ "$(basename "${TRAIN_LIST}")" != *test* && "$(basename "${VAL_LIST}")" != *test* ]] || {
    echo "TEST split is forbidden" >&2; exit 4;
  }
  echo "RGB-anchored change superpixel V3 preflight passed. Test access: FORBIDDEN."
}

train_stage() {
  local stage="$1" steps="$2" eval_every="$3" lr="$4" out="$5" init_ckpt="${6:-}"
  mkdir -p "${out}" "${LOG_ROOT}"
  local -a args resume init
  mapfile -t args < <(common_args)
  resume=(); init=()
  if [[ -s "${out}/latest.pth" ]]; then
    resume+=(--resume "${out}/latest.pth")
  elif [[ -n "${init_ckpt}" ]]; then
    test -s "${init_ckpt}" || { echo "Missing previous stage: ${init_ckpt}" >&2; exit 5; }
    init+=(--init_ckpt "${init_ckpt}")
  fi
  python -u -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.train \
    "${args[@]}" --stage "${stage}" --output_dir "${out}" \
    --optimizer_steps "${steps}" --eval_every "${eval_every}" --save_every "${SAVE_EVERY:-2000}" \
    --val_max_batches "${VAL_MAX_BATCHES:-0}" --plugin_lr "${lr}" "${init[@]}" "${resume[@]}"
}

audit_fixed() {
  local kind="$1"
  mkdir -p "${AUDIT_ROOT}"
  python -u -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.audit_partition \
    --mode "${kind}" --variant p2_rgb_structure --a1_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --output_json "${AUDIT_ROOT}/p${kind}_partition.json" \
    --batch_size "${AUDIT_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" --device "${DEVICE}"
}

audit_learned() {
  local ckpt="${CHECKPOINT:-${STAGE1_OUT}/best.pth}"
  [[ -s "${ckpt}" ]] || ckpt="${STAGE1_OUT}/latest.pth"
  test -s "${ckpt}" || { echo "Missing learned checkpoint" >&2; exit 6; }
  mkdir -p "${AUDIT_ROOT}"
  python -u -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.audit_partition \
    --mode learned --checkpoint "${ckpt}" --a1_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --output_json "${AUDIT_ROOT}/${VARIANT}_partition.json" \
    --batch_size "${AUDIT_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" --device "${DEVICE}"
}

diagnose_change() {
  local ckpt="${CHECKPOINT:-${STAGE1_OUT}/best.pth}"
  [[ -s "${ckpt}" ]] || ckpt="${STAGE1_OUT}/latest.pth"
  test -s "${ckpt}" || { echo "Missing learned checkpoint" >&2; exit 6; }
  mkdir -p "${AUDIT_ROOT}"
  python -u -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.diagnose_change_branch \
    --checkpoint "${ckpt}" --variant "${VARIANT}" --a1_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --output_json "${AUDIT_ROOT}/${VARIANT}_change_diagnostic.json" \
    --batch_size "${AUDIT_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-4}" \
    --max_batches "${DIAGNOSTIC_MAX_BATCHES:-64}" \
    --gradient_batches "${DIAGNOSTIC_GRADIENT_BATCHES:-8}" \
    --shift_batches "${DIAGNOSTIC_SHIFT_BATCHES:-8}" \
    --device "${DEVICE}" --seed "${SEED}"
}

summarize_change() {
  local json_path="${AUDIT_ROOT}/${VARIANT}_change_diagnostic.json"
  test -s "${json_path}" || { echo "Missing change diagnostic: ${json_path}" >&2; exit 7; }
  python -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.summarize_change_diagnostic \
    "${json_path}"
}

visualize_compare() {
  local ckpt="${CHECKPOINT:-${STAGE1_OUT}/best.pth}"
  [[ -s "${ckpt}" ]] || ckpt="${STAGE1_OUT}/latest.pth"
  test -s "${ckpt}" || { echo "Missing learned checkpoint" >&2; exit 6; }
  local visual_root="${VISUAL_OUTPUT:-${AUDIT_ROOT}/visual_compare}"
  local -a sample_args
  sample_args=()
  if [[ -n "${SAMPLE_ID:-}" ]]; then
    sample_args+=(--sample_id "${SAMPLE_ID}")
  fi
  python -u -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.visualize_partition_compare \
    --checkpoint "${ckpt}" --a1_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --output_dir "${visual_root}" --focus_class "${FOCUS_CLASS:-2}" \
    --device "${DEVICE}" "${sample_args[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    VAL_MAX_BATCHES=8 train_stage structure 20 10 "${STAGE0_LR:-1e-4}" "${OUTPUT_ROOT}/_smoke/${VARIANT}_stage0" ""
    ;;
  stage0) preflight; train_stage structure "${STAGE0_STEPS:-1000}" "${STAGE0_EVAL_EVERY:-500}" "${STAGE0_LR:-1e-4}" "${STAGE0_OUT}" "" ;;
  stage1) preflight; train_stage change "${STAGE1_STEPS:-3000}" "${STAGE1_EVAL_EVERY:-500}" "${STAGE1_LR:-1e-4}" "${STAGE1_OUT}" "${STAGE0_OUT}/latest.pth" ;;
  pipeline)
    preflight
    train_stage structure "${STAGE0_STEPS:-1000}" "${STAGE0_EVAL_EVERY:-500}" "${STAGE0_LR:-1e-4}" "${STAGE0_OUT}" ""
    train_stage change "${STAGE1_STEPS:-3000}" "${STAGE1_EVAL_EVERY:-500}" "${STAGE1_LR:-1e-4}" "${STAGE1_OUT}" "${STAGE0_OUT}/latest.pth"
    ;;
  audit_grid) preflight; audit_fixed grid ;;
  audit_slico) preflight; audit_fixed slico ;;
  audit_learned) preflight; audit_learned ;;
  diagnose_change) preflight; diagnose_change ;;
  summarize_change) summarize_change ;;
  visualize) preflight; visualize_compare ;;
  status)
    pgrep -af 'damageformer_rgb_anchored_change_superpixel_v3.train' || true
    find "${BASE_OUT}" -name training_summary.json -print 2>/dev/null || true
    ;;
  *) echo "Usage: $0 {preflight|smoke|stage0|stage1|pipeline|audit_grid|audit_slico|audit_learned|diagnose_change|summarize_change|visualize|status} [variant]" >&2; exit 2 ;;
esac
