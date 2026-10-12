#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit) ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|smoke|audit" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
CVPRW26_ROOT="${CVPRW26_ROOT:-${REPO_ROOT}/cvprw26}"
VAL_LIST="${VAL_LIST:-${CVPRW26_ROOT}/data/splits/val_set.txt}"
INSTANCE_JSON_DIR="${INSTANCE_JSON_DIR:-${DATASET_ROOT}/target_instance_level}"
INSTANCE_COCO_JSON="${INSTANCE_COCO_JSON:-}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
CHECKPOINT="${CHECKPOINT:-${P3_CKPT}}"
MODEL_TAG="${MODEL_TAG:-p3_locked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_MICB_BAG_AUDIT}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_SPI_MICB_BAG_AUDIT.sh
  python -m py_compile script/standard_ML/spi_t1/bag_audit.py
  python -m unittest tests.spi_t1.test_bag_audit -v
  echo "SPI MICB Bag audit preflight passed."
  exit 0
fi

for required in "${DATASET_ROOT}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${CHECKPOINT}"; do
  test -e "${required}" || { echo "Missing MICB Bag audit input: ${required}" >&2; exit 1; }
done
observed_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ -n "${EXPECTED_CHECKPOINT_SHA256}" && "${observed_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint identity mismatch: expected=${EXPECTED_CHECKPOINT_SHA256}, observed=${observed_sha256}" >&2
  exit 1
fi

annotation_args=()
if [[ -n "${INSTANCE_COCO_JSON}" ]]; then
  annotation_args+=(--instance_coco_json "${INSTANCE_COCO_JSON}")
else
  test -d "${INSTANCE_JSON_DIR}" || { echo "Missing instance directory: ${INSTANCE_JSON_DIR}" >&2; exit 1; }
  annotation_args+=(--instance_json_dir "${INSTANCE_JSON_DIR}")
fi

mkdir -p "${OUTPUT_ROOT}"
max_images=0
tag=validation_full
if [[ "${MODE}" == "smoke" ]]; then
  max_images="${SMOKE_IMAGES:-2}"
  tag="smoke${max_images}"
fi

python -u -m bda_benchmark.script.standard_ML.spi_t1.bag_audit \
  --checkpoint "${CHECKPOINT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_list "${VAL_LIST}" \
  --sar_stats_json "${SAR_STATS_JSON}" \
  "${annotation_args[@]}" \
  --output_json "${OUTPUT_ROOT}/SPI_MICB_BAG_${MODEL_TAG}_${tag}_seed3407.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-4}" \
  --minimum_area "${MINIMUM_AREA:-32}" \
  --minimum_purity "${MINIMUM_PURITY:-0.5}" \
  --expected_images "${EXPECTED_IMAGES:-697}" \
  --max_images "${max_images}" \
  --progress_every "${PROGRESS_EVERY:-10}"
