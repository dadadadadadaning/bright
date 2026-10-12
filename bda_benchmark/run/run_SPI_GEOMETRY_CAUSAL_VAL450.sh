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
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
RGID_CKPT="${RGID_CKPT:-}"
RGID_PROVENANCE_JSON="${RGID_PROVENANCE_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_GEOMETRY_CAUSAL_VAL450}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh
  python -m py_compile \
    script/standard_ML/spi_geometry_causal/core.py \
    script/standard_ML/spi_geometry_causal/audit.py
  python -m unittest tests.spi_geometry_causal.test_core -v
  echo "SPI locked-A6 geometry causal preflight passed."
  exit 0
fi

for required in "${DATASET_ROOT}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}"; do
  test -e "${required}" || { echo "Missing geometry causal input: ${required}" >&2; exit 1; }
done
if [[ -n "${RGID_CKPT}" || -n "${RGID_PROVENANCE_JSON}" ]]; then
  test -s "${RGID_CKPT}" || { echo "Missing RGID checkpoint: ${RGID_CKPT}" >&2; exit 1; }
  test -s "${RGID_PROVENANCE_JSON}" || { echo "Missing RGID provenance: ${RGID_PROVENANCE_JSON}" >&2; exit 1; }
fi

observed_sha256="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
if [[ -n "${EXPECTED_CHECKPOINT_SHA256}" && "${observed_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "P3 identity mismatch: expected=${EXPECTED_CHECKPOINT_SHA256}, observed=${observed_sha256}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
max_images=0
tag="validation_full_controls_only_v1"
bootstrap_iterations="${BOOTSTRAP_ITERATIONS:-10000}"
if [[ -n "${RGID_CKPT}" ]]; then
  tag="validation_full_with_rgid_g3_v1"
fi
if [[ "${MODE}" == "smoke" ]]; then
  max_images="${SMOKE_IMAGES:-2}"
  tag="smoke${max_images}_${tag}"
  bootstrap_iterations=10
fi

rgid_args=()
if [[ -n "${RGID_CKPT}" ]]; then
  rgid_args+=(--rgid_checkpoint "${RGID_CKPT}")
  rgid_args+=(--rgid_provenance_json "${RGID_PROVENANCE_JSON}")
fi

python -u -m bda_benchmark.script.standard_ML.spi_geometry_causal.audit \
  --checkpoint "${P3_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_list "${VAL_LIST}" \
  --sar_stats_json "${SAR_STATS_JSON}" \
  --output_json "${OUTPUT_ROOT}/SPI_GEOMETRY_CAUSAL_${tag}_seed3407.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-4}" \
  --minimum_area 32 \
  --local_kernel_size 5 \
  --a6_minimum_purity 0.70 \
  --a6_minimum_margin 0.10 \
  --expected_images 450 \
  --max_images "${max_images}" \
  --progress_every "${PROGRESS_EVERY:-10}" \
  --bootstrap_iterations "${bootstrap_iterations}" \
  --seed 3407 \
  --expected_p3_miou 0.605020 \
  --reproduction_tolerance 0.00001 \
  "${rgid_args[@]}"

