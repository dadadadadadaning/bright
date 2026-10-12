#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|calibration|audit|auto) ;;
  *) echo "Choose preflight|smoke|calibration|audit|auto" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n "${SCRIPT_DIR}/run.sh"
  python -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/audit.py"
  python -m unittest discover -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
  echo "SPI Structural Diffusion V1 isolated preflight passed"
  exit 0
fi

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_STRUCTURAL_DIFFUSION_V1}"
EXPECTED_SHA256="${EXPECTED_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

if [[ "${MODE}" == "auto" ]]; then
  bash "${SCRIPT_DIR}/run.sh" calibration
  calibration_json="${OUTPUT_ROOT}/SPI_STRUCTURAL_DIFFUSION_internal_calibration_v1_seed3407.json"
  decision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "${calibration_json}")"
  if [[ "${decision}" != "PROMOTE_STRUCTURAL_DIFFUSION" ]]; then
    echo "Automatic stop after internal calibration: ${decision}"
    echo "Val450 was not accessed by this experiment."
    exit 0
  fi
  echo "Internal calibration passed; starting the locked Val450 audit."
  bash "${SCRIPT_DIR}/run.sh" audit
  exit 0
fi

for required in "${DATASET_ROOT}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
[[ -z "${EXPECTED_SHA256}" || "${observed}" == "${EXPECTED_SHA256}" ]] || {
  echo "P3 identity mismatch: ${observed}" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
mkdir -p "${OUTPUT_ROOT}"
max_images=0
tag="validation_full_v1"
iterations="${BOOTSTRAP_ITERATIONS:-10000}"
expected_images="${EXPECTED_IMAGES:-450}"
expected_p3_miou="${EXPECTED_P3_MIOU:-0.605020}"
if [[ "${MODE}" == "smoke" ]]; then
  smoke_list="${CALIBRATION_LIST:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_RISK_GATE_V1/seed3407/split_seed3407/gate_calibration.txt}"
  if [[ -s "${smoke_list}" ]]; then
    VAL_LIST="${smoke_list}"
  fi
  max_images="${SMOKE_IMAGES:-10}"
  tag="smoke${max_images}_v1"
  iterations=20
elif [[ "${MODE}" == "calibration" ]]; then
  VAL_LIST="${CALIBRATION_LIST:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_RISK_GATE_V1/seed3407/split_seed3407/gate_calibration.txt}"
  test -s "${VAL_LIST}" || { echo "Missing calibration list: ${VAL_LIST}" >&2; exit 1; }
  expected_images="${CALIBRATION_IMAGES:-291}"
  expected_p3_miou="-1"
  tag="internal_calibration_v1"
fi

python -u -m bda_benchmark.experiments.spi_structural_diffusion_v1.audit \
  --checkpoint "${P3_CKPT}" \
  --dataset-path "${DATASET_ROOT}" \
  --val-list "${VAL_LIST}" \
  --sar-stats-json "${SAR_STATS_JSON}" \
  --output-json "${OUTPUT_ROOT}/SPI_STRUCTURAL_DIFFUSION_${tag}_seed3407.json" \
  --device cuda:0 \
  --num-workers "${NUM_WORKERS:-4}" \
  --expected-images "${expected_images}" \
  --max-images "${max_images}" \
  --minimum-area "${MINIMUM_AREA:-32}" \
  --progress-every "${PROGRESS_EVERY:-10}" \
  --profile-timing "${PROFILE_TIMING:-1}" \
  --bootstrap-iterations "${iterations}" \
  --seed 3407 \
  --expected-p3-miou "${expected_p3_miou}"
