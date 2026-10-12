#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|train_t0|train_t1|train_pair|audit|full|summarize) ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|smoke|train_t0|train_t1|train_pair|audit|full|summarize" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
if [[ -d /root/data ]]; then DEFAULT_STORAGE_ROOT=/root/data; else DEFAULT_STORAGE_ROOT=/root/shared-nvme; fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
CVPRW26_ROOT="${CVPRW26_ROOT:-${REPO_ROOT}/cvprw26}"
VAL_LIST="${VAL_LIST:-${CVPRW26_ROOT}/data/splits/val_set.txt}"
INSTANCE_JSON_DIR="${INSTANCE_JSON_DIR:-${DATASET_ROOT}/target_instance_level}"
INSTANCE_COCO_JSON="${INSTANCE_COCO_JSON:-}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_T1_ISOLATED}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_T1_ISOLATED}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
SEED="${SEED:-3407}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_SPI_T1_isolated.sh
  python -c "import cv2, imageio, matplotlib, scipy, torch; from pycocotools import mask; from dataset.make_data_loader import MultimodalDamageAssessmentDatset; print('SPI T1 training/evaluation imports OK | cv2=' + cv2.__version__ + ' | matplotlib=' + matplotlib.__version__ + ' | torch=' + torch.__version__)" || {
    echo "Missing SPI T1 dependency or the full BRIGHT training data loader cannot be imported in the active environment." >&2
    echo "Headless-server packages normally required here include opencv-python-headless and matplotlib." >&2
    exit 1
  }
  python -m py_compile \
    script/standard_ML/spi_t1/core.py \
    script/standard_ML/spi_t1/train.py \
    script/standard_ML/spi_t1/audit.py \
    script/standard_ML/spi_t1/summarize.py
  python -m unittest tests.spi_t1.test_core tests.spi_t1.test_pair_contract -v
  echo "SPI T0/T1 isolated preflight passed."
  exit 0
fi

for required in "${DATASET_ROOT}" "${TRAIN_LIST}" "${P3_CKPT}"; do
  [[ -e "${required}" ]] || { echo "Missing SPI T1 input: ${required}" >&2; exit 1; }
done

train_variant() {
  local variant="$1"
  local steps="$2"
  local out="${OUTPUT_ROOT}/seed${SEED}/${variant}"
  python -u -m bda_benchmark.script.standard_ML.spi_t1.train \
    --variant "${variant}" \
    --dataset_path "${DATASET_ROOT}" \
    --train_list "${TRAIN_LIST}" \
    --base_checkpoint "${P3_CKPT}" \
    --expected_base_sha256 "${EXPECTED_P3_SHA256}" \
    --output_dir "${out}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --crop_size "${CROP_SIZE:-640}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
    --optimizer_steps "${steps}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --minimum_component_area "${MINIMUM_COMPONENT_AREA:-32}" \
    --lambda_cb "${LAMBDA_CB:-0.10}" \
    --log_every "${LOG_EVERY:-20}" \
    --save_every "${SAVE_EVERY:-500}"
}

audit_pair() {
  local maximum_batches="$1"
  for required in \
    "${OUTPUT_ROOT}/seed${SEED}/t0_original_objective/last.pth" \
    "${OUTPUT_ROOT}/seed${SEED}/t1_component_balanced/last.pth" \
    "${VAL_LIST}" "${SAR_STATS_JSON}"; do
    test -s "${required}" || { echo "Missing SPI T1 audit input: ${required}" >&2; exit 1; }
  done
  local annotation_args=()
  if [[ -n "${INSTANCE_COCO_JSON}" ]]; then
    annotation_args+=(--instance_coco_json "${INSTANCE_COCO_JSON}")
  else
    test -d "${INSTANCE_JSON_DIR}" || { echo "Missing instance directory: ${INSTANCE_JSON_DIR}" >&2; exit 1; }
    annotation_args+=(--instance_json_dir "${INSTANCE_JSON_DIR}")
  fi
  mkdir -p "${AUDIT_ROOT}"
  local tag=validation_full_dual_gt
  if [[ "${maximum_batches}" != "0" ]]; then tag="smoke${maximum_batches}_dual_gt"; fi
  python -u -m bda_benchmark.script.standard_ML.spi_t1.audit \
    --base_checkpoint "${P3_CKPT}" \
    --t0_checkpoint "${OUTPUT_ROOT}/seed${SEED}/t0_original_objective/last.pth" \
    --t1_checkpoint "${OUTPUT_ROOT}/seed${SEED}/t1_component_balanced/last.pth" \
    --dataset_path "${DATASET_ROOT}" \
    --val_list "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    "${annotation_args[@]}" \
    --output_json "${AUDIT_ROOT}/SPI_T1_${tag}_seed${SEED}.json" \
    --output_event_tsv "${AUDIT_ROOT}/SPI_T1_${tag}_seed${SEED}.events.tsv" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-8}" \
    --expected_images "${EXPECTED_EVALUATION_IMAGES:-697}" \
    --max_batches "${maximum_batches}" \
    --progress_every "${PROGRESS_EVERY:-10}"
}

case "${MODE}" in
  smoke)
    train_variant t0_original_objective "${SMOKE_STEPS:-2}"
    train_variant t1_component_balanced "${SMOKE_STEPS:-2}"
    audit_pair "${SMOKE_BATCHES:-2}"
    ;;
  train_pair)
    train_variant t0_original_objective "${OPTIMIZER_STEPS:-3000}"
    train_variant t1_component_balanced "${OPTIMIZER_STEPS:-3000}"
    ;;
  train_t0)
    train_variant t0_original_objective "${OPTIMIZER_STEPS:-3000}"
    ;;
  train_t1)
    train_variant t1_component_balanced "${OPTIMIZER_STEPS:-3000}"
    ;;
  audit)
    audit_pair 0
    ;;
  full)
    train_variant t0_original_objective "${OPTIMIZER_STEPS:-3000}"
    train_variant t1_component_balanced "${OPTIMIZER_STEPS:-3000}"
    audit_pair 0
    ;;
  summarize)
    python -u -m bda_benchmark.script.standard_ML.spi_t1.summarize \
      --inputs \
      "${AUDIT_ROOT}/SPI_T1_validation_full_dual_gt_seed3407.json" \
      "${AUDIT_ROOT}/SPI_T1_validation_full_dual_gt_seed4407.json" \
      "${AUDIT_ROOT}/SPI_T1_validation_full_dual_gt_seed5407.json" \
      --output "${AUDIT_ROOT}/SPI_T1_cross_seed_summary.json"
    ;;
esac
