#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
REGION="${2:-r2_slico}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

case "${REGION}" in r1_grid|r2_slico|r3_shuffled|r4_learned) ;; *) echo "bad region: ${REGION}" >&2; exit 2;; esac

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_ALIGNED_REGION_MEMORY_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_ALIGNED_REGION_MEMORY_V1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/SPI_ALIGNED_REGION_MEMORY_V1}"
STEPS="${STEPS:-5000}"
SEED="${SEED:-3407}"
LEARNED_CKPT="${LEARNED_CKPT:-${OUTPUT_ROOT}/r4_learned_tokenizer_step${STEPS}_seed${SEED}.pth}"

train_records() { echo "${OUTPUT_ROOT}/${1}_train_records.pt"; }
val_records() { echo "${OUTPUT_ROOT}/${1}_val450_records.pt"; }
metric_path() { echo "${OUTPUT_ROOT}/${1}_aligned_key_step${STEPS}.pth"; }
memory_path() { echo "${OUTPUT_ROOT}/${1}_train_memory_step${STEPS}.pt"; }
direct_path() { echo "${RESULT_ROOT}/${1}_direct_step${STEPS}.json"; }
audit_path() { echo "${RESULT_ROOT}/${1}_cross_event_retrieval_step${STEPS}.json"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/core.py" "${HERE}/train_regions.py" "${HERE}/build_records.py" "${HERE}/audit_direct.py" \
    "${HERE}/compare_regions.py" "${HERE}/admit_memory.py" \
    "${BDA_ROOT}/model/DamageFormer_ROI_LCA.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  test -s "${ROI_CKPT}" || { echo "Missing admitted ROI-LCA checkpoint: ${ROI_CKPT}" >&2; exit 3; }
  test -s "${SAR_STATS_JSON}" || { echo "Missing train-only SAR statistics: ${SAR_STATS_JSON}" >&2; exit 3; }
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "Aligned Region Memory V1 preflight passed. Test access: FORBIDDEN."
}

train_learned_regions() {
  mkdir -p "${OUTPUT_ROOT}"
  [[ -s "${LEARNED_CKPT}" && "${FORCE:-0}" != 1 ]] && { echo "Learned tokenizer exists: ${LEARNED_CKPT}"; return; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.train_regions \
    --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" --output "${LEARNED_CKPT}" \
    --steps "${STEPS}" --batch_size "${REGION_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED}"
}

build_records() {
  local split="$1" region="$2" list output extra=()
  if [[ "${split}" == train ]]; then list="${TRAIN_LIST}"; output="$(train_records "${region}")"; else list="${VAL_LIST}"; output="$(val_records "${region}")"; fi
  [[ -s "${output}" && "${FORCE:-0}" != 1 ]] && { echo "Records exist: ${output}"; return; }
  [[ "${region}" == r4_learned ]] && extra+=(--region_checkpoint "${LEARNED_CKPT}")
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --split "${split}" \
    --roi_variant "${ROI_VARIANT}" --roi_checkpoint "${ROI_CKPT}" --sar_stats_json "${SAR_STATS_JSON}" \
    --region_variant "${region}" --output "${output}" --segments "${SEGMENTS:-400}" \
    --compactness "${COMPACTNESS:-10}" --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}" "${extra[@]}"
}

stage_region() {
  local region="$1" train val metric direct
  train="$(train_records "${region}")"; val="$(val_records "${region}")"
  metric="$(metric_path "${region}")"; direct="$(direct_path "${region}")"
  build_records train "${region}"; build_records validation "${region}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.train_key \
    --records "${train}" --output "${metric}" --variant t3_multiring --steps "${STEPS}" \
    --batch_size "${METRIC_BATCH_SIZE:-96}" --device "${DEVICE:-cuda:0}" --seed "${SEED}"
  mkdir -p "${RESULT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.audit_direct \
    --records "${val}" --metric "${metric}" --output_json "${direct}" --device "${DEVICE:-cuda:0}"
}

stage_memory() {
  local region="$1" train val metric memory audit
  train="$(train_records "${region}")"; val="$(val_records "${region}")"
  metric="$(metric_path "${region}")"; memory="$(memory_path "${region}")"; audit="$(audit_path "${region}")"
  test -s "${train}" && test -s "${val}" && test -s "${metric}" || {
    echo "Run the region stage first: ${region}" >&2; exit 5; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.build_memory \
    --records "${train}" --metric "${metric}" --output "${memory}" \
    --per_class_event "${MEMORY_PER_CLASS_EVENT:-128}" --hard_fraction "${HARD_FRACTION:-0.30}" \
    --device "${DEVICE:-cuda:0}" --seed "${SEED}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.audit_retrieval \
    --query_records "${val}" --metric "${metric}" --memory "${memory}" --output_json "${audit}" \
    --topk "${TOPK:-8}" --max_per_event "${MAX_PER_EVENT:-2}" --geometry_weight 0 \
    --exclude_same_image --exclude_same_event --shuffle_repeats 10 \
    --device "${DEVICE:-cuda:0}" --seed "${SEED}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.admit_memory \
    --audit "${audit}" --output "${RESULT_ROOT}/${region}_memory_admission_step${STEPS}.json"
}

compare_fixed() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.compare_regions \
    --grid "$(direct_path r1_grid)" --slico "$(direct_path r2_slico)" \
    --shuffled "$(direct_path r3_shuffled)" --output "${RESULT_ROOT}/fixed_region_admission_step${STEPS}.json"
}

compare_all() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.compare_regions \
    --grid "$(direct_path r1_grid)" --slico "$(direct_path r2_slico)" \
    --shuffled "$(direct_path r3_shuffled)" --learned "$(direct_path r4_learned)" \
    --output "${RESULT_ROOT}/all_region_admission_step${STEPS}.json"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight; FORCE=1 BATCH_SIZE=1 NUM_WORKERS=0 \
      "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_aligned_region_memory_v1.build_records \
      --dataset_path "${DATASET_PATH}" --data_list "${VAL_LIST}" --split validation \
      --roi_variant "${ROI_VARIANT}" --roi_checkpoint "${ROI_CKPT}" --sar_stats_json "${SAR_STATS_JSON}" \
      --region_variant r2_slico --output "${OUTPUT_ROOT}/smoke2_records.pt" --max_images 2 --num_workers 0 --device "${DEVICE:-cuda:0}" ;;
  train_regions) preflight; train_learned_regions ;;
  build_train) preflight; build_records train "${REGION}" ;;
  build_val) preflight; build_records validation "${REGION}" ;;
  stage) preflight; stage_region "${REGION}" ;;
  memory) preflight; stage_memory "${REGION}" ;;
  stage_fixed)
    preflight
    for value in r1_grid r2_slico r3_shuffled; do echo "===== ${value} START ====="; stage_region "${value}"; done
    compare_fixed ;;
  stage_learned) preflight; train_learned_regions; stage_region r4_learned; compare_all ;;
  compare_fixed) compare_fixed ;;
  compare_all) compare_all ;;
  writeback)
    echo "Writeback is deliberately locked in V1 until region admission and Memory>Direct are both proven." >&2
    echo "Do not bypass this guard; use the later V2 Utility-Verifier stage after admission." >&2
    exit 6 ;;
  *) echo "usage: $0 preflight|smoke|train_regions|build_train|build_val|stage|memory|stage_fixed|stage_learned|compare_fixed|compare_all|writeback [region]" >&2; exit 2 ;;
esac
