#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit|train|auto) ;;
  *) echo "Choose preflight|smoke|audit|train|auto" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_cluster_detail_d1_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "Cluster Detail D1 Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" \
    "${SCRIPT_DIR}/audit_e0.py" "${SCRIPT_DIR}/train_verifier.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Cluster Detail D1 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
D0_JSON="${D0_JSON:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_DETAIL_D0_V1/cluster_detail_d0_seed${SEED}.json}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_DETAIL_D1_V1}"
TRAIN_CACHE="${CACHE_ROOT}/meta_train.pt"
CAL_CACHE="${CACHE_ROOT}/meta_calibration.pt"
ADMISSION_JSON="${RESULT_ROOT}/cluster_detail_d1_e0_seed${SEED}.json"
VERIFIER_DIR="${CACHE_ROOT}/verifier_step${VERIFIER_STEPS:-1000}"

for path in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${D0_JSON}"; do
  test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${D0_JSON}')); assert p.get('decision') == 'ADMIT_CLUSTER_DETAIL_D1_STAGE2_EDGE_EVIDENCE'; print('Cluster Detail D0 prerequisite passed')"
mkdir -p "${CACHE_ROOT}" "${RESULT_ROOT}"
TRAIN_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split="$1"; local count="$2"; local maximum="$3"; local output="$4"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_detail_d1_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split}" --expected-images "${count}" \
    --max-images "${maximum}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-1}" --minimum-building-area "${MINIMUM_BUILDING_AREA:-32}" \
    --minimum-cluster-area "${MINIMUM_CLUSTER_AREA:-33}" \
    --feature-channels "${FEATURE_CHANNELS:-32}" --matching-window "${MATCHING_WINDOW:-5}" \
    --halo-radius "${HALO_RADIUS:-3}" --progress-every "${PROGRESS_EVERY:-10}" \
    --checkpoint-every "${STATE_EVERY:-50}" --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${TRAIN_COUNT}" "${SMOKE_IMAGES:-20}" "${CACHE_ROOT}/smoke.pt"
  echo "Cluster Detail D1 smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

run_audit() {
  [[ -s "${TRAIN_CACHE}" ]] || build_cache meta_train "${TRAIN_COUNT}" 0 "${TRAIN_CACHE}"
  [[ -s "${CAL_CACHE}" ]] || build_cache meta_calibration "${CAL_COUNT}" 0 "${CAL_CACHE}"
  if [[ ! -s "${ADMISSION_JSON}" ]]; then
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_detail_d1_v1.audit_e0 \
      --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
      --d0-json "${D0_JSON}" --output "${ADMISSION_JSON}" \
      --utility-margin "${UTILITY_MARGIN:-0.10}" --permutations "${PERMUTATIONS:-1000}" \
      --seed "${SEED}"
  fi
}

run_train() {
  test -s "${ADMISSION_JSON}" || { echo "Run audit first: missing ${ADMISSION_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${ADMISSION_JSON}')); assert p.get('decision') == 'ADMIT_CLUSTER_DETAIL_D1_E1_SINGLE_VERIFIER', p.get('decision'); print('D1 E0 admission passed')"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_detail_d1_v1.train_verifier \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --admission-json "${ADMISSION_JSON}" --output-dir "${VERIFIER_DIR}" \
    --steps "${VERIFIER_STEPS:-1000}" --hidden "${VERIFIER_HIDDEN:-32}" \
    --batch-size "${VERIFIER_BATCH_SIZE:-2048}" --learning-rate "${LEARNING_RATE:-1e-3}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
}

case "${MODE}" in
  audit) run_audit ;;
  train) run_train ;;
  auto)
    run_audit
    decision="$(${PYTHON_BIN} -c "import json; print(json.load(open('${ADMISSION_JSON}'))['decision'])")"
    if [[ "${decision}" == "ADMIT_CLUSTER_DETAIL_D1_E1_SINGLE_VERIFIER" ]]; then
      run_train
    else
      echo "Automatic stop after D1 E0: ${decision}. Verifier was not trained."
    fi
    ;;
esac
