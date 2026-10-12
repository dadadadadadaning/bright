#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/p_sc_residual_seed3407/stage1_change/latest.pth}"
TRAIN_CACHE="${TRAIN_CACHE:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_train_all_buildings.pt}"
VAL_CACHE="${VAL_CACHE:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_val_all_buildings.pt}"
OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_PROTOTYPE_MEMORY_VALIDATION_V1}"
EVAL_ROOT="${EVAL_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_PROTOTYPE_MEMORY_VALIDATION_V1}"
IMAGE_CMS="${OUT_ROOT}/val_image_pixel_cms.pt"
RESULT="${EVAL_ROOT}/prototype_controls_event_bootstrap.json"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3407}"

preflight() {
  python -m py_compile experiments/damageformer_prototype_memory_validation_v1/*.py
  python -m unittest experiments.damageformer_prototype_memory_validation_v1.test_contract -v
  for path in "${VAL_LIST}" "${A1_CKPT}" "${STAGE1_CKPT}" "${TRAIN_CACHE}" "${VAL_CACHE}"; do
    test -s "${path}" || { echo "Missing required file: ${path}" >&2; exit 3; }
  done
  [[ "$(basename "${VAL_LIST}")" != *test* ]] || { echo "TEST access forbidden" >&2; exit 4; }
  echo "Prototype Memory Validation V1 preflight passed. Coverage locked=2%. Test access: FORBIDDEN."
}

build_images() {
  mkdir -p "${OUT_ROOT}"
  if [[ -s "${IMAGE_CMS}" && "${FORCE:-0}" != "1" ]]; then
    echo "Reuse ${IMAGE_CMS}"; return 0
  fi
  python -u -m bda_benchmark.experiments.damageformer_prototype_memory_validation_v1.build_image_cms \
    --val_cache "${VAL_CACHE}" --checkpoint "${STAGE1_CKPT}" --a1_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" --output "${IMAGE_CMS}" \
    --device "${DEVICE}" --batch_size "${BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}"
}

validate() {
  test -s "${IMAGE_CMS}" || { echo "Run build_images first" >&2; exit 5; }
  mkdir -p "${EVAL_ROOT}"
  python -u -m bda_benchmark.experiments.damageformer_prototype_memory_validation_v1.validate \
    --train_cache "${TRAIN_CACHE}" --val_cache "${VAL_CACHE}" --image_cms "${IMAGE_CMS}" \
    --output "${RESULT}" --coverage 0.02 --multi_k "${MULTI_K:-4}" \
    --control_repeats "${CONTROL_REPEATS:-10}" --bootstrap_iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  build_images) preflight; build_images ;;
  validate) preflight; validate ;;
  summarize) python -m bda_benchmark.experiments.damageformer_prototype_memory_validation_v1.summarize --input "${RESULT}" ;;
  all) preflight; build_images; validate ;;
  *) echo "Usage: $0 {preflight|build_images|validate|summarize|all}" >&2; exit 2 ;;
esac

