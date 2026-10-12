#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-p_sc_residual}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${VARIANT}" in
  p_s_struct_only|p_c_change_only|p_sc_residual|p_sc_no_peak|p_sc_mean_only|p_sc_no_bag|p_sc_no_uncertainty|p_sc_shuffled_sar) ;;
  *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2}"
SEED="${SEED:-3407}"
DEVICE="${DEVICE:-cuda:0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TARGET_REGION_AREA="${TARGET_REGION_AREA:-16}"
MIN_REGION_AREA="${MIN_REGION_AREA:-4}"
MAX_REGIONS_PER_BUILDING="${MAX_REGIONS_PER_BUILDING:-16}"
BUILDING_THRESHOLD="${BUILDING_THRESHOLD:-0.50}"
MIN_BUILDING_AREA="${MIN_BUILDING_AREA:-8}"
CHANGE_ALPHA_MAX="${CHANGE_ALPHA_MAX:-0.30}"
PEAK_TEMPERATURE="${PEAK_TEMPERATURE:-0.20}"
BAG_AGGREGATOR="${BAG_AGGREGATOR:-logmeanexp}"
BAG_TEMPERATURE="${BAG_TEMPERATURE:-0.50}"
BAG_TOPK="${BAG_TOPK:-3}"
BAG_LABEL_PURITY="${BAG_LABEL_PURITY:-0.70}"

BASE_OUT="${OUTPUT_ROOT}/${VARIANT}_seed${SEED}"
STAGE0_OUT="${BASE_OUT}/stage0_structure"
STAGE1_OUT="${BASE_OUT}/stage1_change"
STAGE2_OUT="${BASE_OUT}/stage2_feedback"

common_args() {
  printf '%s\n' \
    --variant "${VARIANT}" \
    --dataset_path "${DATASET_PATH}" \
    --train_list "${TRAIN_LIST}" \
    --val_list "${VAL_LIST}" \
    --a1_ckpt "${A1_CKPT}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --device "${DEVICE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --seed "${SEED}" \
    --target_region_area "${TARGET_REGION_AREA}" \
    --min_region_area "${MIN_REGION_AREA}" \
    --max_regions_per_building "${MAX_REGIONS_PER_BUILDING}" \
    --building_threshold "${BUILDING_THRESHOLD}" \
    --min_building_area "${MIN_BUILDING_AREA}" \
    --change_alpha_max "${CHANGE_ALPHA_MAX}" \
    --peak_temperature "${PEAK_TEMPERATURE}" \
    --bag_aggregator "${BAG_AGGREGATOR}" \
    --bag_temperature "${BAG_TEMPERATURE}" \
    --bag_topk "${BAG_TOPK}" \
    --bag_label_purity "${BAG_LABEL_PURITY}"
}

preflight() {
  python -m py_compile experiments/damageformer_building_internal_change_partition_v2/*.py
  python -m unittest experiments.damageformer_building_internal_change_partition_v2.test_contract -v
  test -s "${A1_CKPT}" || { echo "Missing A1 checkpoint: ${A1_CKPT}" >&2; exit 3; }
  test -s "${TRAIN_LIST}" || { echo "Missing train list: ${TRAIN_LIST}" >&2; exit 3; }
  test -s "${VAL_LIST}" || { echo "Missing val list: ${VAL_LIST}" >&2; exit 3; }
  [[ "$(basename "${TRAIN_LIST}")" != *test* && "$(basename "${VAL_LIST}")" != *test* ]] || {
    echo "TEST split is forbidden" >&2; exit 4;
  }
  python - <<'PY'
import torch
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import DamageFormerBuildingInternalChangePartitionV2
print('V2 model import: OK')
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
  echo "PREFLIGHT PASSED"
}

exact_replay() {
  mkdir -p "${AUDIT_ROOT}/${VARIANT}/exact_replay"
  python -u -m bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.exact_replay \
    --a1_ckpt "${A1_CKPT}" --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --variant "${VARIANT}" --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE}" --target_region_area "${TARGET_REGION_AREA}" \
    --min_region_area "${MIN_REGION_AREA}" --max_regions_per_building "${MAX_REGIONS_PER_BUILDING}" \
    --output_json "${AUDIT_ROOT}/${VARIANT}/exact_replay/result.json"
}

train_stage() {
  local stage="$1" steps="$2" eval_every="$3" lr="$4" out="$5" init_ckpt="${6:-}" vmax="${7:-0}"
  mkdir -p "${out}" "${LOG_ROOT}"
  local -a args resume init
  mapfile -t args < <(common_args)
  resume=(); init=()
  if [[ -s "${out}/latest.pth" ]]; then
    resume+=(--resume "${out}/latest.pth")
  elif [[ -n "${init_ckpt}" ]]; then
    test -s "${init_ckpt}" || { echo "Missing previous-stage checkpoint: ${init_ckpt}" >&2; exit 5; }
    init+=(--init_ckpt "${init_ckpt}")
  fi
  python -u -m bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train \
    "${args[@]}" --stage "${stage}" --output_dir "${out}" \
    --optimizer_steps "${steps}" --eval_every "${eval_every}" --save_every "${SAVE_EVERY:-2000}" \
    --val_max_batches "${vmax}" --plugin_lr "${lr}" "${init[@]}" "${resume[@]}"
}

stage0() {
  train_stage structure "${STAGE0_STEPS:-1000}" "${STAGE0_EVAL_EVERY:-500}" "${STAGE0_LR:-1e-4}" "${STAGE0_OUT}" "" "${VAL_MAX_BATCHES:-0}"
}

stage1() {
  local init="${STAGE0_INIT_CKPT:-${STAGE0_OUT}/latest.pth}"
  train_stage change "${STAGE1_STEPS:-3000}" "${STAGE1_EVAL_EVERY:-500}" "${STAGE1_LR:-1e-4}" "${STAGE1_OUT}" "${init}" "${VAL_MAX_BATCHES:-0}"
}

stage2() {
  local default_init="${STAGE1_OUT}/best.pth"
  [[ -s "${default_init}" ]] || default_init="${STAGE1_OUT}/latest.pth"
  local init="${STAGE1_INIT_CKPT:-${default_init}}"
  train_stage feedback "${STAGE2_STEPS:-5000}" "${STAGE2_EVAL_EVERY:-500}" "${STAGE2_LR:-5e-5}" "${STAGE2_OUT}" "${init}" "${VAL_MAX_BATCHES:-0}"
}

smoke_stage() {
  local stage="$1" smoke_out="${OUTPUT_ROOT}/_smoke/${VARIANT}_${stage}_seed${SEED}" init=""
  rm -rf "${smoke_out}"
  case "${stage}" in
    structure) ;;
    change)
      init="${STAGE0_INIT_CKPT:-${STAGE0_OUT}/latest.pth}"
      ;;
    feedback)
      init="${STAGE1_INIT_CKPT:-${STAGE1_OUT}/latest.pth}"
      ;;
  esac
  train_stage "${stage}" 20 10 "${SMOKE_LR:-1e-4}" "${smoke_out}" "${init}" 8
}

audit_ckpt() {
  local ckpt="$1" tag="$2"
  test -s "${ckpt}" || { echo "Missing checkpoint: ${ckpt}" >&2; exit 6; }
  mkdir -p "${AUDIT_ROOT}/${VARIANT}"
  python -u -m bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.audit \
    --checkpoint "${ckpt}" --a1_ckpt "${A1_CKPT}" --dataset_path "${DATASET_PATH}" \
    --val_list "${VAL_LIST}" --batch_size "${AUDIT_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" \
    --device "${DEVICE}" --output_json "${AUDIT_ROOT}/${VARIANT}/${tag}.json"
}

case "${MODE}" in
  preflight) preflight ;;
  exact_replay) preflight; exact_replay ;;
  smoke0) preflight; exact_replay; smoke_stage structure ;;
  smoke1) preflight; smoke_stage change ;;
  smoke2) preflight; smoke_stage feedback ;;
  stage0) preflight; stage0 ;;
  stage1) preflight; stage1 ;;
  stage2) preflight; stage2 ;;
  pipeline) preflight; exact_replay; stage0; stage1; stage2 ;;
  audit0) preflight; audit_ckpt "${STAGE0_OUT}/latest.pth" stage0_structure ;;
  audit1)
    preflight
    ckpt="${STAGE1_OUT}/best.pth"; [[ -s "${ckpt}" ]] || ckpt="${STAGE1_OUT}/latest.pth"
    audit_ckpt "${ckpt}" stage1_change
    ;;
  audit2)
    preflight
    ckpt="${STAGE2_OUT}/best.pth"; [[ -s "${ckpt}" ]] || ckpt="${STAGE2_OUT}/latest.pth"
    audit_ckpt "${ckpt}" stage2_feedback
    ;;
  audit_custom)
    preflight
    test -n "${CHECKPOINT:-}" || { echo "Set CHECKPOINT=/path/to/file.pth" >&2; exit 2; }
    audit_ckpt "${CHECKPOINT}" "${AUDIT_TAG:-custom}"
    ;;
  status)
    pgrep -af 'damageformer_building_internal_change_partition_v2.train' || true
    find "${BASE_OUT}" -maxdepth 3 -name training_summary.json -print 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|exact_replay|smoke0|smoke1|smoke2|stage0|stage1|stage2|pipeline|audit0|audit1|audit2|audit_custom|status} [variant]" >&2
    exit 2
    ;;
esac
