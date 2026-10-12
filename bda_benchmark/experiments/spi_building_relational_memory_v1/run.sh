#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-m6_aligned_relation_quality}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

case "${VARIANT}" in
  m0_rgb_mean|m1_unaligned_multistat|m2_aligned_multistat|m3_aligned_multiring|m4_aligned_relation|m5_shuffled_relation|m6_aligned_relation_quality) ;;
  *) echo "bad variant: ${VARIANT}" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
ROI_VARIANT="${ROI_VARIANT:-a3_conf3}"
ROI_TAG="${ROI_TAG:-probe5000}"
ROI_ROOT="${ROI_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_ROI_LCA_V1}"
ROI_CKPT="${ROI_CKPT:-${ROI_ROOT}/${ROI_VARIANT}_${ROI_TAG}_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_BUILDING_RELATIONAL_MEMORY_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_BUILDING_RELATIONAL_MEMORY_V1}"
STEPS="${STEPS:-5000}"
SEED="${SEED:-3407}"

flat_path() { echo "${OUTPUT_ROOT}/flat_slico_${ROI_VARIANT}_${1}.pt"; }
building_path() { echo "${OUTPUT_ROOT}/buildings_${ROI_VARIANT}_${1}.pt"; }
model_path() { echo "${OUTPUT_ROOT}/${1}_step${STEPS}_seed${SEED}.pth"; }
result_path() { echo "${RESULT_ROOT}/${1}_step${STEPS}.json"; }
SIZE_BINS="${OUTPUT_ROOT}/train_size_bins.json"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/core.py" "${HERE}/build_flat_records.py" \
    "${HERE}/build_buildings.py" "${HERE}/train.py" \
    "${HERE}/audit.py" "${HERE}/summarize.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  test -s "${ROI_CKPT}" || { echo "Missing admitted ROI-LCA checkpoint: ${ROI_CKPT}" >&2; exit 3; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing train-only SAR stats: ${SAR_STATS_JSON}" >&2; exit 3; }
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "Building Relational Memory V1 preflight passed. Test access: FORBIDDEN."
}

build_flat() {
  local split="$1" list output
  if [[ "${split}" == train ]]; then list="${TRAIN_LIST}"; else list="${VAL_LIST}"; fi
  output="$(flat_path "${split}")"
  [[ -s "${output}" && "${FORCE:-0}" != 1 ]] && { echo "Flat records exist: ${output}"; return; }
  mkdir -p "${OUTPUT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_relational_memory_v1.build_flat_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" \
    --split "$([[ "${split}" == train ]] && echo train || echo validation)" \
    --roi_variant "${ROI_VARIANT}" --roi_checkpoint "${ROI_CKPT}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --output "${output}" --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
    --batch_size "${BUILD_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}"
}

build_buildings() {
  local split="$1" output="$(building_path "$1")"
  [[ -s "${output}" && "${FORCE:-0}" != 1 ]] && { echo "Building records exist: ${output}"; return; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_relational_memory_v1.build_buildings \
    --region_records "$(flat_path "${split}")" --split "${split}" --output "${output}" \
    --size_bins_json "${SIZE_BINS}" --max_tokens "${MAX_TOKENS:-24}" \
    --min_tokens "${MIN_TOKENS:-2}" --building_purity "${BUILDING_PURITY:-0.80}"
}

prepare() {
  build_flat train; build_flat validation
  build_buildings train; build_buildings validation
}

train_one() {
  local variant="$1" output="$(model_path "$1")"
  [[ -s "${output}" && "${FORCE:-0}" != 1 ]] && { echo "Model exists: ${output}"; return; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_relational_memory_v1.train \
    --records "$(building_path train)" --variant "${variant}" --output "${output}" \
    --steps "${STEPS}" --batch_size "${TRAIN_BATCH_SIZE:-48}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED}"
}

audit_one() {
  local variant="$1" output="$(result_path "$1")"
  mkdir -p "${RESULT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_relational_memory_v1.audit \
    --train_records "$(building_path train)" --val_records "$(building_path validation)" \
    --checkpoint "$(model_path "${variant}")" --output_json "${output}" \
    --per_class_event_size "${MEMORY_PER_BUCKET:-64}" --coarse_topr "${COARSE_TOPR:-32}" \
    --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" --shuffle_repeats 10 \
    --exclude_same_event --batch_size "${AUDIT_BATCH_SIZE:-128}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED}"
}

summarize() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_relational_memory_v1.summarize \
    --result_root "${RESULT_ROOT}" --steps "${STEPS}" \
    --output_json "${RESULT_ROOT}/summary_step${STEPS}.json"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) preflight ;;
  prepare) preflight; prepare ;;
  train) preflight; prepare; train_one "${VARIANT}" ;;
  audit) preflight; prepare; test -s "$(model_path "${VARIANT}")"; audit_one "${VARIANT}" ;;
  stage) preflight; prepare; train_one "${VARIANT}"; audit_one "${VARIANT}" ;;
  stage_all)
    preflight; prepare
    for value in m0_rgb_mean m1_unaligned_multistat m2_aligned_multistat m3_aligned_multiring m4_aligned_relation m5_shuffled_relation m6_aligned_relation_quality; do
      echo "===== ${value} START ====="; train_one "${value}"; audit_one "${value}"; echo "===== ${value} COMPLETE ====="
    done
    summarize ;;
  summarize) summarize ;;
  writeback)
    echo "Pixel writeback is locked until aligned evidence, correct relations, Memory>Direct, and recovery gates all pass." >&2
    exit 6 ;;
  *) echo "usage: $0 preflight|smoke|prepare|train|audit|stage|stage_all|summarize|writeback [variant]" >&2; exit 2 ;;
esac
