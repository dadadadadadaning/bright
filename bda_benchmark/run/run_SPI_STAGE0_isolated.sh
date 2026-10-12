#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
case "${MODE}" in
  preflight|smoke|audit) ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|audit" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
CVPRW26_ROOT="${CVPRW26_ROOT:-${REPO_ROOT}/cvprw26}"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
if [[ -d /root/data ]]; then
  DEFAULT_STORAGE_ROOT=/root/data
else
  DEFAULT_STORAGE_ROOT=/root/shared-nvme
fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${CVPRW26_ROOT}/data/splits/val_set.txt}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
A2_CKPT="${A2_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
INSTANCE_JSON_DIR="${INSTANCE_JSON_DIR:-${DATASET_ROOT}/target_instance_level}"
DEFAULT_INSTANCE_COCO_JSON="${CVPRW26_ROOT}/data/instance_annotations/val.json"
INSTANCE_COCO_JSON="${INSTANCE_COCO_JSON:-}"
RGID_PREDICTIONS_JSON="${RGID_PREDICTIONS_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_STAGE0_ISOLATED}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
EXPECTED_EVALUATION_IMAGES="${EXPECTED_EVALUATION_IMAGES:-697}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODULE="bda_benchmark.script.standard_ML.spi_stage0.audit"
TEST_MODULES=("tests.spi_stage0.test_core" "tests.spi_stage0.test_annotations")

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_SPI_STAGE0_isolated.sh
  python -m py_compile \
    script/standard_ML/spi_stage0/annotations.py \
    script/standard_ML/spi_stage0/core.py \
    script/standard_ML/spi_stage0/audit.py
  if ! python -c "from pycocotools import mask as _mask"; then
    echo "Missing dependency: pycocotools" >&2
    exit 1
  fi
  python -m unittest -v "${TEST_MODULES[@]}"
  echo "SPI Stage-0 isolated preflight passed."
  exit 0
fi

for required in "${P3_CKPT}" "${A2_CKPT}" "${SAR_STATS_JSON}" "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing SPI Stage-0 input: ${required}" >&2
    exit 1
  }
done

FIRST_SAMPLE="$(awk 'NF {print $1; exit}' "${VAL_LIST}")"
if [[ -z "${FIRST_SAMPLE}" ]]; then
  echo "SPI Stage-0 validation list is empty: ${VAL_LIST}" >&2
  exit 1
fi
for required_pair in \
  "${DATASET_ROOT}/pre-event/${FIRST_SAMPLE}_pre_disaster.tif" \
  "${DATASET_ROOT}/post-event/${FIRST_SAMPLE}_post_disaster.tif"; do
  test -s "${required_pair}" || {
    echo "DATASET_ROOT does not cover the Stage-0 public validation split." >&2
    echo "Missing first validation pair member: ${required_pair}" >&2
    echo "Set DATASET_ROOT to the BRIGHT root containing pre-event/, post-event/, and target_instance_level/." >&2
    exit 1
  }
done

annotation_args=()
if [[ -z "${INSTANCE_COCO_JSON}" && -s "${DEFAULT_INSTANCE_COCO_JSON}" ]]; then
  INSTANCE_COCO_JSON="${DEFAULT_INSTANCE_COCO_JSON}"
fi
if [[ -n "${INSTANCE_COCO_JSON}" ]]; then
  test -s "${INSTANCE_COCO_JSON}" || {
    echo "Missing unified instance annotations: ${INSTANCE_COCO_JSON}" >&2
    exit 1
  }
  annotation_args+=(--instance_coco_json "${INSTANCE_COCO_JSON}")
else
  test -d "${INSTANCE_JSON_DIR}" || {
    echo "Missing official per-image annotations: ${INSTANCE_JSON_DIR}" >&2
    exit 1
  }
  annotation_args+=(--instance_json_dir "${INSTANCE_JSON_DIR}")
fi

rgid_args=()
if [[ -n "${RGID_PREDICTIONS_JSON}" ]]; then
  test -s "${RGID_PREDICTIONS_JSON}" || {
    echo "Missing RGID COCO predictions: ${RGID_PREDICTIONS_JSON}" >&2
    exit 1
  }
  if [[ -z "${INSTANCE_COCO_JSON}" ]]; then
    echo "RGID predictions require INSTANCE_COCO_JSON from the same public split." >&2
    exit 1
  fi
  rgid_args+=(--rgid_predictions_json "${RGID_PREDICTIONS_JSON}")
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ "${MODE}" == "smoke" ]]; then
  MAX_BATCHES="${MAX_BATCHES:-2}"
  TAG="smoke${MAX_BATCHES}"
else
  MAX_BATCHES=0
  TAG="validation_full"
fi
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/SPI_STAGE0_${TAG}_seed${SEED:-3407}.json}"
OUTPUT_EVENT_TSV="${OUTPUT_EVENT_TSV:-${OUTPUT_ROOT}/SPI_STAGE0_${TAG}_seed${SEED:-3407}.events.tsv}"

echo "========================================================================================================"
echo "SPI Stage-0 | same CVPRW26 public validation split | no training"
echo "P3 checkpoint     : ${P3_CKPT}"
echo "A2 checkpoint     : ${A2_CKPT}"
echo "Instance labels   : ${INSTANCE_COCO_JSON:-${INSTANCE_JSON_DIR}}"
echo "RGID predictions  : ${RGID_PREDICTIONS_JSON:-not supplied; E3 will be omitted}"
echo "Output            : ${OUTPUT_JSON}"
echo "Test accessed     : NEVER"
echo "========================================================================================================"

python -u -m "${MODULE}" \
  --p3_checkpoint "${P3_CKPT}" \
  --a2_checkpoint "${A2_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  "${annotation_args[@]}" \
  "${rgid_args[@]}" \
  --output_json "${OUTPUT_JSON}" \
  --output_event_tsv "${OUTPUT_EVENT_TSV}" \
  --device cuda:0 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --minimum_gt_area "${MINIMUM_GT_AREA:-1}" \
  --minimum_component_area "${MINIMUM_COMPONENT_AREA:-32}" \
  --rgid_score_threshold "${RGID_SCORE_THRESHOLD:-0.05}" \
  --rgid_mask_nms_iou "${RGID_MASK_NMS_IOU:-0.70}" \
  --match_iou "${MATCH_IOU:-0.50}" \
  --max_batches "${MAX_BATCHES}" \
  --progress_every "${PROGRESS_EVERY:-10}" \
  --expected_p3_sha256 "${EXPECTED_P3_SHA256}" \
  --expected_evaluation_images "${EXPECTED_EVALUATION_IMAGES}" \
  --minimum_valid_miou_gain_pp "${MINIMUM_VALID_MIOU_GAIN_PP:-0.5}" \
  --minimum_building_macro_f1_gain_pp "${MINIMUM_BUILDING_MACRO_F1_GAIN_PP:-1.0}" \
  --minimum_damaged_f1_gain_pp "${MINIMUM_DAMAGED_F1_GAIN_PP:-1.0}" \
  --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
  --maximum_intact_iou_drop_pp "${MAXIMUM_INTACT_IOU_DROP_PP:-0.5}" \
  --seed "${SEED:-3407}"
