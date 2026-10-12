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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_cluster_oracle_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "Cluster Oracle Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "S2 Cluster Oracle preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_CLUSTER_ORACLE_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_CLUSTER_ORACLE_V1}"
RGID_STOP_JSON="${RGID_STOP_JSON:-/root/data/BRIGHT-eval-results/SPI_SELECTIVE_STRIP_RGID_E0_V1_FAST2/selective_rgid_e0_seed${SEED}.json}"

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" \
  "${SPLIT_ROOT}/meta_calibration.txt" "${RGID_STOP_JSON}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${RGID_STOP_JSON}')); assert p.get('decision') == 'STOP_SELECTIVE_RGID_SPLIT_ORACLE_CEILING_LOW'; print('RGID low-ceiling prerequisite passed')"
mkdir -p "${CACHE_ROOT}" "${RESULT_ROOT}"
TRAIN_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split="$1"; local expected="$2"; local output="$3"; local maximum="$4"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_oracle_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split}" --expected-images "${expected}" \
    --max-images "${maximum}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-2}" --minimum-area "${MINIMUM_AREA:-32}" \
    --progress-every "${PROGRESS_EVERY:-20}" \
    --state-every "${STATE_EVERY:-50}" --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  PROGRESS_EVERY="${SMOKE_PROGRESS_EVERY:-1}" \
    build_cache meta_train "${TRAIN_COUNT}" "${CACHE_ROOT}/smoke.pt" "${SMOKE_IMAGES:-10}"
  echo "S2 Cluster Oracle smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

CAL_CACHE="${CACHE_ROOT}/meta_calibration.pt"
RESULT_JSON="${RESULT_ROOT}/cluster_oracle_e0_seed${SEED}.json"
[[ -s "${CAL_CACHE}" ]] || build_cache meta_calibration "${CAL_COUNT}" "${CAL_CACHE}" 0
test ! -e "${RESULT_JSON}" || { echo "Result already exists; refusing repeated calibration audit: ${RESULT_JSON}" >&2; exit 1; }
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_oracle_v1.audit \
  --calibration-cache "${CAL_CACHE}" --output "${RESULT_JSON}" \
  --minimum-gain-vs-building-pp "${MINIMUM_GAIN_VS_BUILDING_PP:-0.10}" \
  --minimum-nonnegative-events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
  --minimum-accepted-clusters "${MINIMUM_ACCEPTED_CLUSTERS:-200}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
echo "S2 Cluster Oracle audit complete. Meta-confirm/Val450/test were not accessed."
