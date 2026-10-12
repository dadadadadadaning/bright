#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-val}"
case "${MODE}" in
  preflight|smoke|val) ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|val" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
CVPRW26_ROOT="${CVPRW26_ROOT:-${REPO_ROOT}/cvprw26}"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${CVPRW26_ROOT}/data/splits/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
A2_CKPT="${A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
INSTANCE_JSON_DIR="${INSTANCE_JSON_DIR:-${DATASET_ROOT}/target_instance_level}"
DEFAULT_INSTANCE_COCO_JSON="${CVPRW26_ROOT}/data/instance_annotations/val.json"
INSTANCE_COCO_JSON="${INSTANCE_COCO_JSON:-}"
DEPLOY_PREDICTIONS_JSON="${DEPLOY_PREDICTIONS_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/H9_building_oracle}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
EXPECTED_EVALUATION_IMAGES="${EXPECTED_EVALUATION_IMAGES:-697}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

ANALYZER="script/standard_ML/audit_H9_building_oracle.py"
TEST_MODULE="tests.test_h9_building_oracle"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_H9_building_oracle.sh
  python -m py_compile "${ANALYZER}"
  if ! python -c "from pycocotools import mask as _mask"; then
    echo "Missing dependency: pycocotools (required to decode official COCO instance masks)." >&2
    echo "Install it in the active bright5090 environment, then rerun preflight." >&2
    exit 1
  fi
  python -m unittest -v "${TEST_MODULE}"
  echo "H9-B0 building-oracle preflight passed."
  exit 0
fi

for required in \
  "${P3_CKPT}" \
  "${A2_CKPT}" \
  "${SAR_STATS_JSON}" \
  "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing H9-B0 input: ${required}" >&2
    exit 1
  }
done

annotation_args=()
if [[ -z "${INSTANCE_COCO_JSON}" && -s "${DEFAULT_INSTANCE_COCO_JSON}" ]]; then
  INSTANCE_COCO_JSON="${DEFAULT_INSTANCE_COCO_JSON}"
fi
if [[ -n "${INSTANCE_COCO_JSON}" ]]; then
  test -s "${INSTANCE_COCO_JSON}" || {
    echo "Missing unified H9 instance annotations: ${INSTANCE_COCO_JSON}" >&2
    exit 1
  }
  annotation_args+=(--instance_coco_json "${INSTANCE_COCO_JSON}")
  annotation_contract="unified COCO: ${INSTANCE_COCO_JSON}"
else
  test -d "${INSTANCE_JSON_DIR}" || {
    echo "Missing official per-image instance annotations: ${INSTANCE_JSON_DIR}" >&2
    echo "Set INSTANCE_JSON_DIR to the directory containing *_instance_damage.json files." >&2
    echo "Download/restore the public CVPRW26 train/val instance-label release before running H9-B0." >&2
    exit 1
  }
  annotation_args+=(--instance_json_dir "${INSTANCE_JSON_DIR}")
  annotation_contract="official per-image COCO: ${INSTANCE_JSON_DIR}"
fi
if [[ -n "${DEPLOY_PREDICTIONS_JSON}" ]]; then
  test -s "${DEPLOY_PREDICTIONS_JSON}" || {
    echo "Missing deploy instance predictions: ${DEPLOY_PREDICTIONS_JSON}" >&2
    exit 1
  }
  if [[ -z "${INSTANCE_COCO_JSON}" ]]; then
    echo "External deploy predictions require INSTANCE_COCO_JSON built for the same CVPRW26 public validation split." >&2
    echo "Without it, leave DEPLOY_PREDICTIONS_JSON empty to use the clearly labelled P3 connected-component proxy." >&2
    exit 1
  fi
fi

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

OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/H9_B0_building_oracle_${TAG}.json}"
OUTPUT_EVENT_TSV="${OUTPUT_EVENT_TSV:-${OUTPUT_ROOT}/H9_B0_building_oracle_${TAG}.events.tsv}"
OUTPUT_DISASTER_TSV="${OUTPUT_DISASTER_TSV:-${OUTPUT_ROOT}/H9_B0_building_oracle_${TAG}.disasters.tsv}"
OUTPUT_CLASS_TSV="${OUTPUT_CLASS_TSV:-${OUTPUT_ROOT}/H9_B0_building_oracle_${TAG}.classes.tsv}"

deploy_args=()
deploy_contract="P3 localization connected-component proxy"
if [[ -n "${DEPLOY_PREDICTIONS_JSON}" ]]; then
  deploy_args+=(--deploy_predictions_json "${DEPLOY_PREDICTIONS_JSON}")
  deploy_contract="external COCO instance predictions"
fi

echo "================================================================================"
echo "Protocol          : H9-B0 validation-only building-level oracle"
echo "Evaluation split  : CVPRW26 public val (${EXPECTED_EVALUATION_IMAGES} images)"
echo "P3                : ${P3_CKPT}"
echo "H6-A2             : ${A2_CKPT}"
echo "Official instances: ${annotation_contract}"
echo "Deploy geometry   : ${deploy_contract}"
echo "Output JSON       : ${OUTPUT_JSON}"
echo "Test accessed     : NEVER"
echo "================================================================================"

python -u "${ANALYZER}" \
  --p3_checkpoint "${P3_CKPT}" \
  --a2_checkpoint "${A2_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  "${annotation_args[@]}" \
  --output_json "${OUTPUT_JSON}" \
  --output_event_tsv "${OUTPUT_EVENT_TSV}" \
  --output_disaster_tsv "${OUTPUT_DISASTER_TSV}" \
  --output_class_tsv "${OUTPUT_CLASS_TSV}" \
  --device cuda:0 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --minimum_gt_area "${MINIMUM_GT_AREA:-1}" \
  --minimum_predicted_area "${MINIMUM_PREDICTED_AREA:-32}" \
  --deploy_score_threshold "${DEPLOY_SCORE_THRESHOLD:-0.05}" \
  --deploy_mask_nms_iou "${DEPLOY_MASK_NMS_IOU:-0.70}" \
  --match_iou "${MATCH_IOU:-0.50}" \
  --max_batches "${MAX_BATCHES}" \
  --expected_p3_sha256 "${EXPECTED_P3_SHA256}" \
  --minimum_macro_f1_gain_pp "${MINIMUM_MACRO_F1_GAIN_PP:-1.0}" \
  --minimum_damaged_f1_gain_pp "${MINIMUM_DAMAGED_F1_GAIN_PP:-1.0}" \
  --minimum_positive_events "${MINIMUM_POSITIVE_EVENTS:-10}" \
  --minimum_deploy_valid_gain_pp "${MINIMUM_DEPLOY_VALID_GAIN_PP:-0.5}" \
  --evaluation_protocol "cvprw26_public_val" \
  --expected_evaluation_images "${EXPECTED_EVALUATION_IMAGES}" \
  --seed "${SEED:-3407}" \
  "${deploy_args[@]}"
