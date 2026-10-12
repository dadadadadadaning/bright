#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit|train|all) ;;
  *) echo "Choose preflight|smoke|audit|train|all" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_modification_pattern_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "SMP Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" \
    "${SCRIPT_DIR}/audit_e0.py" "${SCRIPT_DIR}/train_verifier.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -c "import scipy,torch,imageio; print('dependencies:', scipy.__version__,torch.__version__,imageio.__version__)"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "SMP preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_MODIFICATION_PATTERN_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_MODIFICATION_PATTERN_V1}"
RGID_STOP_JSON="${RGID_STOP_JSON:-/root/data/BRIGHT-eval-results/SPI_SELECTIVE_STRIP_RGID_E0_V1_FAST2/selective_rgid_e0_seed${SEED}.json}"
TRAIN_CACHE="${OUTPUT_ROOT}/meta_train.pt"
CAL_CACHE="${OUTPUT_ROOT}/meta_calibration.pt"
E0_JSON="${RESULT_ROOT}/smp_e0_seed${SEED}.json"

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" "${SPLIT_ROOT}/meta_calibration.txt"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
TRAIN_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split="$1"; local list="$2"; local count="$3"; local output="$4"; local maximum="$5"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_modification_pattern_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" --list-path "${list}" \
    --sar-stats-json "${SAR_STATS_JSON}" --output "${output}" --split "${split}" \
    --expected-images "${count}" --max-images "${maximum}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-2}" --minimum-area "${MINIMUM_AREA:-32}" \
    --state-every "${STATE_EVERY:-100}" --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${OUTPUT_ROOT}/smoke.pt" "${SMOKE_IMAGES:-20}"
  echo "SMP real-data smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

if [[ "${MODE}" == "audit" || "${MODE}" == "all" ]]; then
  test -s "${RGID_STOP_JSON}" || { echo "Missing completed RGID stop result: ${RGID_STOP_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${RGID_STOP_JSON}')); assert p.get('decision') == 'STOP_SELECTIVE_RGID_SPLIT_ORACLE_CEILING_LOW'; print('RGID low-ceiling prerequisite passed')"
  [[ -s "${TRAIN_CACHE}" ]] || build_cache meta_train "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${TRAIN_CACHE}" 0
  [[ -s "${CAL_CACHE}" ]] || build_cache meta_calibration "${SPLIT_ROOT}/meta_calibration.txt" "${CAL_COUNT}" "${CAL_CACHE}" 0
  test ! -e "${E0_JSON}" || { echo "E0 result already exists; refusing repeated meta_calibration audit: ${E0_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_modification_pattern_v1.audit_e0 \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" --output "${E0_JSON}" \
    --utility-margin "${UTILITY_MARGIN:-0.10}" --permutations "${PERMUTATIONS:-1000}" --seed "${SEED}"
  [[ "${MODE}" == "audit" ]] && exit 0
fi

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  test -s "${TRAIN_CACHE}" || { echo "Run audit first: missing ${TRAIN_CACHE}" >&2; exit 1; }
  test -s "${E0_JSON}" || { echo "Run audit first: missing ${E0_JSON}" >&2; exit 1; }
  test ! -e "${OUTPUT_ROOT}/verifier_e1/verifier_frozen.pth" || { echo "Frozen verifier already exists; refusing retraining" >&2; exit 1; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_modification_pattern_v1.train_verifier \
    --train-cache "${TRAIN_CACHE}" --admission-json "${E0_JSON}" \
    --output-dir "${OUTPUT_ROOT}/verifier_e1" --steps "${VERIFIER_STEPS:-1000}" \
    --hidden "${VERIFIER_HIDDEN:-32}" --utility-margin "${UTILITY_MARGIN:-0.10}" --seed "${SEED}"
fi

