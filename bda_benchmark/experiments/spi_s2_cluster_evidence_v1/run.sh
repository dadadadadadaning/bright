#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit|train|confirm) ;;
  *) echo "Choose preflight|smoke|audit|train|confirm" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_cluster_evidence_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "Cluster Evidence Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" \
    "${SCRIPT_DIR}/audit_e0.py" "${SCRIPT_DIR}/train_verifier.py" \
    "${SCRIPT_DIR}/evaluate_confirm.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -c 'import scipy, torch; print("dependencies:", scipy.__version__, torch.__version__)'
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "S2 Cluster Evidence preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_CLUSTER_EVIDENCE_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_CLUSTER_EVIDENCE_V1}"
ORACLE_JSON="${ORACLE_JSON:-/root/data/BRIGHT-eval-results/SPI_S2_CLUSTER_ORACLE_V1_FAST/cluster_oracle_e0_seed${SEED}.json}"
SMP_STOP_JSON="${SMP_STOP_JSON:-/root/data/BRIGHT-eval-results/SPI_S2_MODIFICATION_PATTERN_V1/smp_e0_seed${SEED}.json}"
RAW_GROUPS="${EVIDENCE_GROUPS:-}"
case "${RAW_GROUPS,,}" in
  ""|0|false|off|none|null)
    GROUPS="spatial,rgb_role,sar_role,direct,matching,relation"
    ;;
  *)
    GROUPS="${RAW_GROUPS}"
    ;;
esac
echo "Cluster evidence groups: ${GROUPS}"

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" \
  "${SPLIT_ROOT}/meta_calibration.txt" "${SPLIT_ROOT}/meta_confirm.txt" \
  "${ORACLE_JSON}" "${SMP_STOP_JSON}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${ORACLE_JSON}')); assert p.get('decision') == 'ADMIT_S2_CLUSTER_EVIDENCE_E0'; print('Cluster Oracle prerequisite passed')"
"${PYTHON_BIN}" -c "import json; p=json.load(open('${SMP_STOP_JSON}')); assert p.get('decision') == 'STOP_SMP_NO_SPATIAL_UTILITY_SIGNAL'; print('SMP negative-control prerequisite passed')"
mkdir -p "${CACHE_ROOT}" "${RESULT_ROOT}"
TRAIN_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"
CONFIRM_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_confirm'])")"

build_cache() {
  local split="$1"; local expected="$2"; local output="$3"; local maximum="$4"
  local cluster_cap=0
  if [[ "${split}" == "meta_train" ]]; then
    cluster_cap="${MAX_TRAIN_CLUSTERS_PER_IMAGE:-128}"
  fi
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_evidence_v1.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/${split}.txt" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split}" --expected-images "${expected}" \
    --max-images "${maximum}" --device "${DEVICE:-cuda:0}" \
    --num-workers "${NUM_WORKERS:-1}" --minimum-area "${MINIMUM_AREA:-32}" \
    --feature-channels "${FEATURE_CHANNELS:-32}" \
    --matching-window "${MATCHING_WINDOW:-5}" --halo-radius "${HALO_RADIUS:-2}" \
    --max-clusters-per-image "${cluster_cap}" \
    --groups "${GROUPS}" --progress-every "${PROGRESS_EVERY:-10}" \
    --state-every "${STATE_EVERY:-50}" --seed "${SEED}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${TRAIN_COUNT}" "${CACHE_ROOT}/smoke.pt" "${SMOKE_IMAGES:-10}"
  echo "Cluster Evidence smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

TRAIN_CACHE="${CACHE_ROOT}/meta_train.pt"
CAL_CACHE="${CACHE_ROOT}/meta_calibration.pt"
ADMISSION_JSON="${RESULT_ROOT}/cluster_evidence_e0_seed${SEED}.json"
VERIFIER_DIR="${CACHE_ROOT}/verifier_e1"
SCREEN_JSON="${VERIFIER_DIR}/calibration_screen.json"
VERIFIER_CKPT="${VERIFIER_DIR}/verifier_frozen.pth"
CONFIRM_CACHE="${CACHE_ROOT}/meta_confirm.pt"
CONFIRM_JSON="${RESULT_ROOT}/cluster_verifier_e1_confirm_seed${SEED}.json"

if [[ "${MODE}" == "audit" ]]; then
  [[ -s "${TRAIN_CACHE}" ]] || build_cache meta_train "${TRAIN_COUNT}" "${TRAIN_CACHE}" 0
  [[ -s "${CAL_CACHE}" ]] || build_cache meta_calibration "${CAL_COUNT}" "${CAL_CACHE}" 0
  test ! -e "${ADMISSION_JSON}" || { echo "Refusing repeated E0 audit: ${ADMISSION_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_evidence_v1.audit_e0 \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --oracle-json "${ORACLE_JSON}" --output "${ADMISSION_JSON}" \
    --utility-margin "${UTILITY_MARGIN:-0.10}" \
    --permutations "${PERMUTATIONS:-1000}" --seed "${SEED}"
  echo "Cluster Evidence E0 audit complete. Meta-confirm/Val450/test were not accessed."
  exit 0
fi

if [[ "${MODE}" == "train" ]]; then
  test -s "${TRAIN_CACHE}" || { echo "Run audit first: missing ${TRAIN_CACHE}" >&2; exit 1; }
  test -s "${CAL_CACHE}" || { echo "Run audit first: missing ${CAL_CACHE}" >&2; exit 1; }
  test -s "${ADMISSION_JSON}" || { echo "Run audit first: missing ${ADMISSION_JSON}" >&2; exit 1; }
  test ! -e "${SCREEN_JSON}" || { echo "Refusing repeated calibration training: ${SCREEN_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_evidence_v1.train_verifier \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --admission-json "${ADMISSION_JSON}" --output-dir "${VERIFIER_DIR}" \
    --hidden "${VERIFIER_HIDDEN:-32}" --steps "${VERIFIER_STEPS:-1000}" \
    --batch-size "${VERIFIER_BATCH_SIZE:-4096}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" --seed "${SEED}"
  echo "Cluster Evidence E1 calibration screening complete. Meta-confirm/Val450/test were not accessed."
  exit 0
fi

test -s "${SCREEN_JSON}" || { echo "Run train first: missing ${SCREEN_JSON}" >&2; exit 1; }
"${PYTHON_BIN}" -c "import json; p=json.load(open('${SCREEN_JSON}')); assert p.get('decision') == 'FREEZE_CLUSTER_E1_FOR_META_CONFIRM'; print('Frozen verifier passed calibration prerequisite')"
test -s "${VERIFIER_CKPT}" || { echo "Missing verifier: ${VERIFIER_CKPT}" >&2; exit 1; }
test ! -e "${CONFIRM_JSON}" || { echo "Refusing repeated meta_confirm evaluation: ${CONFIRM_JSON}" >&2; exit 1; }
[[ -s "${CONFIRM_CACHE}" ]] || build_cache meta_confirm "${CONFIRM_COUNT}" "${CONFIRM_CACHE}" 0
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_s2_cluster_evidence_v1.evaluate_confirm \
  --confirm-cache "${CONFIRM_CACHE}" --verifier-checkpoint "${VERIFIER_CKPT}" \
  --calibration-screen "${SCREEN_JSON}" --output "${CONFIRM_JSON}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
echo "One-time Cluster Evidence E1 meta-confirm evaluation complete. Val450/test were not accessed."
