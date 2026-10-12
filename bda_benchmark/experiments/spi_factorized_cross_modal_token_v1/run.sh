#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-t2_aligned}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1}"
TRAIN_RECORDS="${TRAIN_RECORDS:-${OUTPUT_ROOT}/train_slico_maximal_records.pt}"
VAL_RECORDS="${VAL_RECORDS:-${OUTPUT_ROOT}/val_slico_maximal_records.pt}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

check_variant() {
  case "$1" in
    t0_concat|t1_factorized|t2_aligned|t3_multiring) ;;
    *) echo "variant must be t0_concat|t1_factorized|t2_aligned|t3_multiring" >&2; exit 2 ;;
  esac
}

metric_path() { echo "${OUTPUT_ROOT}/${1}_key_step${STEPS:-5000}.pth"; }
memory_path() { echo "${OUTPUT_ROOT}/${1}_memory_step${STEPS:-5000}.pt"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/core.py" "${HERE}/build_records.py" "${HERE}/train_key.py" \
    "${HERE}/build_memory.py" "${HERE}/audit_retrieval.py" \
    "${HERE}/audit_complementarity.py" "${HERE}/summarize.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  "${PYTHON_BIN}" - <<'PY'
import skimage
print("scikit-image", skimage.__version__)
PY
  test -s "${PARENT_CKPT}" || { echo "Missing P3 checkpoint: ${PARENT_CKPT}" >&2; exit 3; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing frozen SAR stats: ${SAR_STATS_JSON}" >&2; exit 3; }
  echo "Factorized Cross-Modal Token V1 preflight passed. Test access: FORBIDDEN."
}

build_records() {
  local list="$1" output="$2"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --parent_ckpt "${PARENT_CKPT}" \
    --sar_stats_json "${SAR_STATS_JSON}" --output "${output}" \
    --region_mode "${REGION_MODE:-slico}" --segments "${SEGMENTS:-400}" \
    --compactness "${COMPACTNESS:-10}" --alignment_radius "${ALIGNMENT_RADIUS:-1}" \
    --purity "${PURITY:-0.80}" --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}"
}

train_variant() {
  local variant="$1" metric
  check_variant "${variant}"; metric="$(metric_path "${variant}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.train_key \
    --records "${TRAIN_RECORDS}" --output "${metric}" --variant "${variant}" \
    --steps "${STEPS:-5000}" --batch_size "${METRIC_BATCH_SIZE:-96}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}"
}

build_variant_memory() {
  local variant="$1" metric memory
  check_variant "${variant}"; metric="$(metric_path "${variant}")"; memory="$(memory_path "${variant}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.build_memory \
    --records "${TRAIN_RECORDS}" --metric "${metric}" --output "${memory}" \
    --per_class_event "${MEMORY_PER_CLASS_EVENT:-128}" --hard_fraction "${HARD_FRACTION:-0.30}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}"
}

audit_variant() {
  local variant="$1" suffix="$2" extra="$3" metric memory output
  check_variant "${variant}"; metric="$(metric_path "${variant}")"; memory="$(memory_path "${variant}")"
  output="${RESULT_ROOT}/${variant}_${suffix}_step${STEPS:-5000}.json"
  mkdir -p "${RESULT_ROOT}"
  local options=(--exclude_same_image)
  [[ "${extra}" == "event" ]] && options+=(--exclude_same_event)
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.audit_retrieval \
    --query_records "${VAL_RECORDS}" --metric "${metric}" --memory "${memory}" \
    --output_json "${output}" --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" \
    --geometry_weight "${GEOMETRY_WEIGHT:-0.0}" --shuffle_repeats "${SHUFFLE_REPEATS:-10}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}" "${options[@]}"
}

stage_variant() {
  local variant="$1"
  test -s "${TRAIN_RECORDS}" || build_records "${TRAIN_LIST}" "${TRAIN_RECORDS}"
  test -s "${VAL_RECORDS}" || build_records "${VAL_LIST}" "${VAL_RECORDS}"
  train_variant "${variant}"
  build_variant_memory "${variant}"
  audit_variant "${variant}" "val" "image"
  audit_variant "${variant}" "val_cross_event" "event"
}

case "${MODE}" in
  preflight) preflight ;;
  build_train_records) preflight; build_records "${TRAIN_LIST}" "${TRAIN_RECORDS}" ;;
  build_val_records) preflight; build_records "${VAL_LIST}" "${VAL_RECORDS}" ;;
  train) preflight; test -s "${TRAIN_RECORDS}"; train_variant "${VARIANT}" ;;
  build_memory) preflight; test -s "${TRAIN_RECORDS}"; build_variant_memory "${VARIANT}" ;;
  audit_val) preflight; test -s "${VAL_RECORDS}"; audit_variant "${VARIANT}" "val" "image" ;;
  audit_val_event) preflight; test -s "${VAL_RECORDS}"; audit_variant "${VARIANT}" "val_cross_event" "event" ;;
  stage) preflight; stage_variant "${VARIANT}" ;;
  stage_all)
    preflight
    for value in t0_concat t1_factorized t2_aligned t3_multiring; do
      echo "===== ${value} START ====="
      stage_variant "${value}"
      echo "===== ${value} COMPLETE ====="
    done
    ;;
  summarize)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.summarize \
      --result_root "${RESULT_ROOT}" --steps "${STEPS:-5000}" \
      --output "${RESULT_ROOT}/t0_t3_summary_step${STEPS:-5000}.json"
    ;;
  audit_complementarity)
    preflight
    variant="t3_multiring"
    metric="$(metric_path "${variant}")"
    memory="$(memory_path "${variant}")"
    test -s "${VAL_RECORDS}" || { echo "Missing Val records: ${VAL_RECORDS}" >&2; exit 4; }
    test -s "${metric}" || { echo "Missing T3 metric: ${metric}" >&2; exit 4; }
    test -s "${memory}" || { echo "Missing T3 Memory: ${memory}" >&2; exit 4; }
    mkdir -p "${RESULT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.audit_complementarity \
      --query_records "${VAL_RECORDS}" --metric "${metric}" --memory "${memory}" \
      --output_json "${RESULT_ROOT}/t3_direct_memory_complementarity_step${STEPS:-5000}.json" \
      --output_pt "${OUTPUT_ROOT}/t3_direct_memory_complementarity_detail_step${STEPS:-5000}.pt" \
      --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" \
      --geometry_weight "${GEOMETRY_WEIGHT:-0.0}" --exclude_same_image --exclude_same_event \
      --batch_size "${AUDIT_BATCH_SIZE:-512}" --device "${DEVICE:-cuda:0}"
    ;;
  *) echo "Choose preflight|build_train_records|build_val_records|train|build_memory|audit_val|audit_val_event|stage|stage_all|summarize|audit_complementarity" >&2; exit 2 ;;
esac
