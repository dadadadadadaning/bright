#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit) ;;
  *) echo "Choose preflight|smoke|audit" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_building_utility_v1"
DEFAULT_PYTHON_BIN="/root/data/conda_envs/bright4090/bin/python"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
"${PYTHON_BIN}" -c "import sys; print('BU-E0 Python:', sys.executable)"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/make_split.py" \
    "${SCRIPT_DIR}/build_cache.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -c "import imageio, scipy, skimage; print('imageio', imageio.__version__, 'scipy', scipy.__version__, 'scikit-image', skimage.__version__)"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "BU-E0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SOURCE_LIST="${SOURCE_LIST:-/root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1/split_seed3407/gate_train.txt}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/cache_seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_BUILDING_UTILITY_V1}"
DEFAULT_EVIDENCE_GROUP_LIST="direct,relation,matching,grid,slic"
case "${EVIDENCE_GROUPS:-}" in
  ""|0|1|false|true|False|True) EVIDENCE_GROUP_LIST="${DEFAULT_EVIDENCE_GROUP_LIST}" ;;
  *) EVIDENCE_GROUP_LIST="${EVIDENCE_GROUPS}" ;;
esac

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" "${SOURCE_LIST}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${SPLIT_ROOT}" "${CACHE_ROOT}" "${RESULT_ROOT}"

"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_utility_v1.make_split \
  --source-list "${SOURCE_LIST}" --output-dir "${SPLIT_ROOT}" --seed "${SEED}"

META_TRAIN_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
META_CALIBRATION_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split_name="$1"
  local expected="$2"
  local output="$3"
  local max_images="$4"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_utility_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split_name}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split_name}" --expected-images "${expected}" \
    --max-images "${max_images}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-4}" --minimum-area "${MINIMUM_AREA:-32}" \
    --matching-max-side "${MATCHING_MAX_SIDE:-80}" \
    --matching-window "${MATCHING_WINDOW:-5}" --cell-size "${CELL_SIZE:-4}" \
    --evidence-groups "${EVIDENCE_GROUP_LIST}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${META_TRAIN_COUNT}" "${CACHE_ROOT}/smoke.pt" "${SMOKE_IMAGES:-10}"
  echo "BU-E0 real-data smoke passed. Val450 was not accessed."
  exit 0
fi

TRAIN_CACHE="${CACHE_ROOT}/meta_train.pt"
CALIBRATION_CACHE="${CACHE_ROOT}/meta_calibration.pt"
if [[ ! -f "${TRAIN_CACHE}" ]]; then
  build_cache meta_train "${META_TRAIN_COUNT}" "${TRAIN_CACHE}" 0
fi
if [[ ! -f "${CALIBRATION_CACHE}" ]]; then
  build_cache meta_calibration "${META_CALIBRATION_COUNT}" "${CALIBRATION_CACHE}" 0
fi
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_building_utility_v1.audit \
  --train-cache "${TRAIN_CACHE}" --calibration-cache "${CALIBRATION_CACHE}" \
  --output "${RESULT_ROOT}/bu_e0_seed${SEED}.json" \
  --utility-margin "${UTILITY_MARGIN:-0.10}" \
  --minimum-oracle-gain-pp "${MINIMUM_ORACLE_GAIN_PP:-0.30}"

echo "BU-E0 audit complete. Val450 was not accessed."
