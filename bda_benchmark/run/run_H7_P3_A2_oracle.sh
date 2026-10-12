#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-val}"
case "${MODE}" in
  preflight|smoke|val|report|transitions) ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|val|report|transitions" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
A2_CKPT="${A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/H7_P3_A2_oracle}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

ANALYZER="script/standard_ML/analyze_H7_P3_A2_oracle.py"
SUMMARIZER="script/standard_ML/summarize_H7_event_complementarity.py"
TRANSITION_SUMMARIZER="script/standard_ML/summarize_H7_pairwise_transitions.py"
TEST_MODULE="tests.test_h7_p3_a2_oracle"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_H7_P3_A2_oracle.sh
  python -m py_compile "${ANALYZER}" "${SUMMARIZER}" "${TRANSITION_SUMMARIZER}"
  python -m unittest -v "${TEST_MODULE}"
  echo "H7 P3/A2 oracle-analysis preflight passed."
  exit 0
fi

if [[ "${MODE}" == "transitions" ]]; then
  SOURCE_JSON="${SOURCE_JSON:-${OUTPUT_ROOT}/P3_A2_oracle_validation_full.json}"
  test -s "${SOURCE_JSON}" || {
    echo "Missing H7 oracle JSON: ${SOURCE_JSON}" >&2
    exit 1
  }
  python -u "${TRANSITION_SUMMARIZER}" \
    --input_json "${SOURCE_JSON}" \
    --output_json "${OUTPUT_ROOT}/P3_A2_pairwise_transitions.json" \
    --output_tsv "${OUTPUT_ROOT}/P3_A2_pairwise_transitions.tsv"
  exit 0
fi

if [[ "${MODE}" == "report" ]]; then
  SOURCE_JSON="${SOURCE_JSON:-${OUTPUT_ROOT}/P3_A2_oracle_validation_full.json}"
  test -s "${SOURCE_JSON}" || {
    echo "Missing H7 oracle JSON: ${SOURCE_JSON}" >&2
    exit 1
  }
  python -u "${SUMMARIZER}" \
    --input_json "${SOURCE_JSON}" \
    --output_json "${OUTPUT_ROOT}/P3_A2_event_disaster_comparison.json" \
    --output_event_tsv "${OUTPUT_ROOT}/P3_A2_event_comparison.tsv" \
    --output_disaster_tsv "${OUTPUT_ROOT}/P3_A2_disaster_type_comparison.tsv"
  exit 0
fi

for required in "${P3_CKPT}" "${A2_CKPT}" "${SAR_STATS_JSON}" "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing H7 oracle-analysis input: ${required}" >&2
    exit 1
  }
done

actual_p3_sha256="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
if [[ "${actual_p3_sha256}" != "${EXPECTED_P3_SHA256}" ]]; then
  echo "P3 SHA256 mismatch." >&2
  echo "Expected: ${EXPECTED_P3_SHA256}" >&2
  echo "Observed: ${actual_p3_sha256}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ "${MODE}" == "smoke" ]]; then
  MAX_BATCHES="${MAX_BATCHES:-2}"
  TAG="smoke${MAX_BATCHES}"
else
  MAX_BATCHES=0
  TAG="validation_full"
fi

OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/P3_A2_oracle_${TAG}.json}"
OUTPUT_TSV="${OUTPUT_TSV:-${OUTPUT_ROOT}/P3_A2_oracle_${TAG}.events.tsv}"

echo "=============================================================================="
echo "Protocol       : H7 validation-only P3/A2 oracle audit"
echo "P3             : ${P3_CKPT}"
echo "H6-A2          : ${A2_CKPT}"
echo "Validation list: ${VAL_LIST}"
echo "Output JSON    : ${OUTPUT_JSON}"
echo "Test accessed  : NEVER"
echo "=============================================================================="

python -u "${ANALYZER}" \
  --p3_checkpoint "${P3_CKPT}" \
  --a2_checkpoint "${A2_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --output_event_tsv "${OUTPUT_TSV}" \
  --device cuda:0 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --minimum_building_area "${MINIMUM_BUILDING_AREA:-32}" \
  --max_batches "${MAX_BATCHES}" \
  --expected_p3_sha256 "${EXPECTED_P3_SHA256}" \
  --minimum_valid_gain_pp "${MINIMUM_VALID_GAIN_PP:-0.5}" \
  --minimum_damaged_gain_pp "${MINIMUM_DAMAGED_GAIN_PP:-1.0}" \
  --minimum_positive_event_fraction "${MINIMUM_POSITIVE_EVENT_FRACTION:-0.5}" \
  --seed "${SEED:-3407}"

if [[ "${MODE}" == "val" ]]; then
  SOURCE_JSON="${OUTPUT_JSON}" \
    bash run/run_H7_P3_A2_oracle.sh report
fi
