#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit) ;;
  train) echo "CD-D0 is parameter-free; training is forbidden until D0 admits D1." >&2; exit 3 ;;
  *) echo "Choose preflight|smoke|audit" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_cluster_detail_d0_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "Cluster Detail D0 Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" \
    "${SCRIPT_DIR}/build_geometry_cache.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Cluster Detail D0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_CLUSTER_EVIDENCE_V2_FAST/seed${SEED}}"
CLUSTER_RESULT_ROOT="${CLUSTER_RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_CLUSTER_EVIDENCE_V2_FAST}"
ORACLE_JSON="${ORACLE_JSON:-/root/data/BRIGHT-eval-results/SPI_S2_CLUSTER_ORACLE_V1_FAST/cluster_oracle_e0_seed${SEED}.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D0_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_DETAIL_D0_V1}"
CAL_CACHE="${CLUSTER_ROOT}/meta_calibration.pt"
EVIDENCE_JSON="${CLUSTER_RESULT_ROOT}/cluster_evidence_e0_seed${SEED}.json"
GEOMETRY_CACHE="${OUTPUT_ROOT}/meta_calibration_geometry.pt"
RESULT_JSON="${RESULT_ROOT}/cluster_detail_d0_seed${SEED}.json"

for path in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_calibration.txt" \
  "${CAL_CACHE}" "${EVIDENCE_JSON}" "${ORACLE_JSON}"; do
  test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${EVIDENCE_JSON}')); assert p.get('decision') == 'STOP_CLUSTER_EVIDENCE_NO_CROSS_EVENT_SIGNAL'; print('Coarse Cluster-E0 stop prerequisite passed')"
mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_geometry() {
  local output="$1"; local maximum="$2"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_detail_d0_v1.build_geometry_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/meta_calibration.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --reference-cache "${CAL_CACHE}" --output "${output}" \
    --expected-images "${CAL_COUNT}" --max-images "${maximum}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --minimum-area "${MINIMUM_AREA:-32}" --progress-every "${PROGRESS_EVERY:-10}" \
    --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_geometry "${OUTPUT_ROOT}/smoke.pt" "${SMOKE_IMAGES:-10}"
  echo "Cluster Detail D0 smoke passed. Meta-confirm/Val450/test were not accessed."
  exit 0
fi

[[ -s "${GEOMETRY_CACHE}" ]] || build_geometry "${GEOMETRY_CACHE}" 0
test ! -e "${RESULT_JSON}" || { echo "Refusing repeated D0 audit: ${RESULT_JSON}" >&2; exit 1; }
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_detail_d0_v1.audit \
  --cluster-cache "${CAL_CACHE}" --geometry-cache "${GEOMETRY_CACHE}" \
  --cluster-evidence-json "${EVIDENCE_JSON}" --cluster-oracle-json "${ORACLE_JSON}" \
  --output "${RESULT_JSON}" --utility-margin "${UTILITY_MARGIN:-0.10}"
echo "Cluster Detail D0 audit complete. Meta-confirm/Val450/test were not accessed."
