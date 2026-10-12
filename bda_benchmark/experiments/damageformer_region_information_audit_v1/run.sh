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
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
REGION_CKPT="${REGION_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/p_sc_residual_seed3407/stage1_change/latest.pth}"
OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_REGION_INFORMATION_AUDIT_V1}"
EVAL_ROOT="${EVAL_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_REGION_INFORMATION_AUDIT_V1}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3407}"

TRAIN_RECORDS="${OUT_ROOT}/train_region_records.pt"
VAL_RECORDS="${OUT_ROOT}/val_region_records.pt"

preflight() {
  python -m py_compile experiments/damageformer_region_information_audit_v1/*.py
  python -m unittest experiments.damageformer_region_information_audit_v1.test_contract -v
  for path in "${TRAIN_LIST}" "${VAL_LIST}" "${A1_CKPT}" "${REGION_CKPT}"; do
    test -s "${path}" || { echo "Missing required file: ${path}" >&2; exit 3; }
  done
  [[ "$(basename "${TRAIN_LIST}")" != *test* && "$(basename "${VAL_LIST}")" != *test* ]] || {
    echo "TEST split is forbidden" >&2; exit 4;
  }
  echo "Region Information Audit V1 preflight passed. Test access: FORBIDDEN."
}

build_one() {
  local split="$1" list="$2" output="$3"
  mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"
  if [[ -s "${output}" && -s "${EVAL_ROOT}/partition_${split}.json" && "${FORCE:-0}" != "1" ]]; then
    echo "Reuse existing records: ${output}"
    echo "Set FORCE=1 to rebuild."
    return 0
  fi
  python -u -m bda_benchmark.experiments.damageformer_region_information_audit_v1.build_records \
    --split "${split}" --source_list "${list}" --dataset_path "${DATASET_PATH}" \
    --a1_ckpt "${A1_CKPT}" --region_ckpt "${REGION_CKPT}" \
    --output "${output}" --audit_json "${EVAL_ROOT}/partition_${split}.json" \
    --batch_size "${BUILD_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" \
    --device "${DEVICE}" --max_images "${MAX_IMAGES:-0}"
}

probes() {
  test -s "${TRAIN_RECORDS}" || { echo "Run build_train first: ${TRAIN_RECORDS}" >&2; exit 5; }
  test -s "${VAL_RECORDS}" || { echo "Run build_val first: ${VAL_RECORDS}" >&2; exit 5; }
  mkdir -p "${EVAL_ROOT}/probes"
  python -u -m bda_benchmark.experiments.damageformer_region_information_audit_v1.train_probes \
    --train_records "${TRAIN_RECORDS}" --val_records "${VAL_RECORDS}" \
    --output_dir "${EVAL_ROOT}/probes" --steps "${PROBE_STEPS:-5000}" \
    --batch_size "${PROBE_BATCH_SIZE:-2048}" --device "${DEVICE}" --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  build_train) preflight; build_one train "${TRAIN_LIST}" "${TRAIN_RECORDS}" ;;
  build_val) preflight; build_one validation "${VAL_LIST}" "${VAL_RECORDS}" ;;
  probes) preflight; probes ;;
  summarize)
    python -m bda_benchmark.experiments.damageformer_region_information_audit_v1.summarize --eval_root "${EVAL_ROOT}"
    ;;
  all) preflight; build_one train "${TRAIN_LIST}" "${TRAIN_RECORDS}"; build_one validation "${VAL_LIST}" "${VAL_RECORDS}"; probes ;;
  smoke)
    preflight
    MAX_IMAGES=4 build_one train "${TRAIN_LIST}" "${OUT_ROOT}/smoke_train.pt"
    MAX_IMAGES=4 build_one validation "${VAL_LIST}" "${OUT_ROOT}/smoke_val.pt"
    TRAIN_RECORDS="${OUT_ROOT}/smoke_train.pt" VAL_RECORDS="${OUT_ROOT}/smoke_val.pt" PROBE_STEPS=20 probes
    ;;
  *) echo "Usage: $0 {preflight|smoke|build_train|build_val|probes|summarize|all}" >&2; exit 2 ;;
esac
