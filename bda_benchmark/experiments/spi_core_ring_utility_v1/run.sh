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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_core_ring_utility_v1"
DEFAULT_PYTHON_BIN="/root/data/conda_envs/bright4090/bin/python"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
"${PYTHON_BIN}" -c "import sys; print('Core-Ring E0 Python:', sys.executable)"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Core-Ring Utility E0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CORE_RING_UTILITY_V1/cache_seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CORE_RING_UTILITY_V1}"
BU_RESULT="${BU_RESULT:-/root/data/BRIGHT-eval-results/SPI_BUILDING_UTILITY_V1/bu_e0_seed${SEED}.json}"
BU_CACHE_ROOT="${BU_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/cache_seed${SEED}}"

for required in \
  "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" \
  "${SPLIT_ROOT}/meta_calibration.txt" "${BU_RESULT}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
for required in "${BU_CACHE_ROOT}/meta_train.pt" "${BU_CACHE_ROOT}/meta_calibration.pt"; do
  test -e "${required}" || { echo "Missing BU-E0 provenance cache: ${required}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c \
  "import hashlib,json,torch; p=json.load(open('${BU_RESULT}')); b=torch.load('${BU_CACHE_ROOT}/meta_train.pt',map_location='cpu',weights_only=False); h=hashlib.sha256(); f=open('${P3_CKPT}','rb'); [h.update(x) for x in iter(lambda:f.read(1048576),b'')]; f.close(); assert p['oracle_passed']; assert not p['validation_accessed']; assert not p['official_instances_accessed']; assert not p['test_accessed']; assert h.hexdigest()==b['p3_sha256']; print('BU-E0 oracle and P3 fingerprint prerequisite passed')"
mkdir -p "${CACHE_ROOT}" "${RESULT_ROOT}"

META_TRAIN_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
META_CALIBRATION_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split_name="$1"
  local expected="$2"
  local output="$3"
  local max_images="$4"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_core_ring_utility_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split_name}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split_name}" --expected-images "${expected}" \
    --max-images "${max_images}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-4}" --minimum-area "${MINIMUM_AREA:-32}" \
    --core-radius "${CORE_RADIUS:-1}" --minimum-core-area "${MINIMUM_CORE_AREA:-4}" \
    --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${META_TRAIN_COUNT}" "${CACHE_ROOT}/smoke.pt" "${SMOKE_IMAGES:-10}"
  echo "Core-Ring Utility E0 real-data smoke passed. Val450 was not accessed."
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
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_core_ring_utility_v1.audit \
  --train-cache "${TRAIN_CACHE}" --calibration-cache "${CALIBRATION_CACHE}" \
  --output "${RESULT_ROOT}/core_ring_e0_seed${SEED}.json" \
  --utility-margin "${UTILITY_MARGIN:-0.10}" \
  --permutations "${PERMUTATIONS:-1000}" --seed "${SEED}"

echo "Core-Ring Utility E0 audit complete. Val450 was not accessed."
