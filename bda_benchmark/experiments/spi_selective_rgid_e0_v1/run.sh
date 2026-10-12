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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_selective_rgid_e0_v1"
DEFAULT_PYTHON_BIN="/root/data/conda_envs/bright4090/bin/python"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
"${PYTHON_BIN}" -c "import sys; print('Selective RGID-E0 Python:', sys.executable)"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/runtime.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -c "import scipy,skimage,imageio; print('dependencies:', scipy.__version__, skimage.__version__, imageio.__version__)"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Selective RGID-E0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
RGID_STEPS="${RGID_STEPS:-500}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_SELECTIVE_RGID_E0_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_SELECTIVE_RGID_E0_V1}"
CORE_RING_RESULT="${CORE_RING_RESULT:-/root/data/BRIGHT-eval-results/SPI_CORE_RING_UTILITY_V1/core_ring_e0_seed${SEED}.json}"
RGID_CKPT_INPUT="${RGID_CKPT:-}"
RGID_PROVENANCE_INPUT="${RGID_PROVENANCE:-}"
if [[ -n "${RGID_CKPT_INPUT}" || -n "${RGID_PROVENANCE_INPUT}" ]]; then
  [[ -n "${RGID_CKPT_INPUT}" && -n "${RGID_PROVENANCE_INPUT}" ]] || {
    echo "RGID_CKPT and RGID_PROVENANCE must be supplied together." >&2
    exit 1
  }
  RGID_DIR="$(dirname "${RGID_CKPT_INPUT}")"
  RGID_CKPT="${RGID_CKPT_INPUT}"
  RGID_PROVENANCE="${RGID_PROVENANCE_INPUT}"
  EXTERNAL_RGID=1
else
  RGID_DIR="${OUTPUT_ROOT}/rgid_meta_train_step${RGID_STEPS}"
  RGID_CKPT="${RGID_DIR}/rgid_step_${RGID_STEPS}_calibrated.pth"
  RGID_PROVENANCE="${RGID_DIR}/provenance.json"
  EXTERNAL_RGID=0
fi
POLICY_JSON="${OUTPUT_ROOT}/risk_policy_train_only.json"

echo "RGID source: $([[ "${EXTERNAL_RGID}" == "1" ]] && echo external || echo managed)"
echo "RGID checkpoint: ${RGID_CKPT}"
echo "RGID provenance: ${RGID_PROVENANCE}"

if [[ "${EXTERNAL_RGID}" == "1" ]]; then
  test -s "${RGID_CKPT}" || {
    echo "External RGID checkpoint is missing or empty: ${RGID_CKPT}" >&2
    exit 1
  }
  test -s "${RGID_PROVENANCE}" || {
    echo "External RGID provenance is missing or empty: ${RGID_PROVENANCE}" >&2
    exit 1
  }
fi

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" \
  "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" \
  "${SPLIT_ROOT}/meta_calibration.txt" "${CORE_RING_RESULT}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
"${PYTHON_BIN}" -c "import json; p=json.load(open('${CORE_RING_RESULT}')); assert p['decision'] in ('STOP_CORE_RING_CEILING_LOW_TRY_RGID_E0','STOP_CORE_RING_NO_PREDICTIVE_SIGNAL_TRY_RGID_E0'); assert not p['validation_accessed']; assert not p['official_instances_accessed']; assert not p['test_accessed']; print('Core-Ring stop decision prerequisite passed:', p['decision'])"
TRAIN_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

train_rgid() {
  local steps="$1"
  local output_dir="$2"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.train_online_rgid \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --train-list "${SPLIT_ROOT}/meta_train.txt" --output-dir "${output_dir}" \
    --device "${DEVICE:-cuda:0}" --steps "${steps}" \
    --batch-size "${BATCH_SIZE:-1}" --accumulation "${ACCUMULATION_STEPS:-4}" \
    --num-workers "${NUM_WORKERS:-4}" --calibration-samples "${CALIBRATION_SAMPLES:-256}" \
    --seed "${SEED}"
}

audit_phase() {
  local phase="$1"
  local list_path="$2"
  local expected="$3"
  local output="$4"
  local max_images="$5"
  shift 5
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_selective_rgid_e0_v1.audit \
    --phase "${phase}" --checkpoint "${P3_CKPT}" \
    --rgid-checkpoint "${RGID_CKPT}" --rgid-provenance "${RGID_PROVENANCE}" \
    --rgid-train-list "${SPLIT_ROOT}/meta_train.txt" \
    --dataset-path "${DATASET_ROOT}" --list-path "${list_path}" \
    --sar-stats-json "${SAR_STATS_JSON}" --output "${output}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-4}" \
    --expected-images "${expected}" --max-images "${max_images}" \
    --minimum-area "${MINIMUM_AREA:-32}" \
    --risk-threshold-grid "${RISK_THRESHOLD_GRID:-0.35,0.45,0.55,0.65,0.75}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}" \
    --state-path "${output}.partial.pt" --checkpoint-every "${STATE_EVERY:-25}" "$@"
}

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_DIR="${OUTPUT_ROOT}/smoke_rgid"
  train_rgid "${SMOKE_RGID_STEPS:-2}" "${SMOKE_DIR}"
  ORIGINAL_CKPT="${RGID_CKPT}"; ORIGINAL_PROV="${RGID_PROVENANCE}"
  RGID_CKPT="${SMOKE_DIR}/rgid_step_${SMOKE_RGID_STEPS:-2}_calibrated.pth"
  RGID_PROVENANCE="${SMOKE_DIR}/provenance.json"
  audit_phase fit "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${OUTPUT_ROOT}/smoke_policy.json" "${SMOKE_IMAGES:-2}"
  echo "Selective RGID-E0 real-data smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

if [[ "${EXTERNAL_RGID}" == "0" && ( ! -f "${RGID_CKPT}" || ! -f "${RGID_PROVENANCE}" ) ]]; then
  echo "Training pseudo-only RGID on meta_train (${RGID_STEPS} optimizer steps)."
  train_rgid "${RGID_STEPS}" "${RGID_DIR}"
fi
if [[ ! -f "${POLICY_JSON}" ]]; then
  echo "Fitting split-risk threshold on meta_train only."
  audit_phase fit "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${POLICY_JSON}" 0
fi
echo "Evaluating the single frozen policy on meta_calibration once."
FINAL_JSON="${RESULT_ROOT}/selective_rgid_e0_seed${SEED}.json"
if [[ -f "${FINAL_JSON}" ]]; then
  echo "Frozen meta_calibration result already exists; refusing to evaluate it a second time: ${FINAL_JSON}"
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${FINAL_JSON}')); print('Decision:', p['decision']); print('Gains:', p.get('gains_miou_pp', {}))"
  exit 0
fi
audit_phase evaluate "${SPLIT_ROOT}/meta_calibration.txt" "${CAL_COUNT}" \
  "${FINAL_JSON}" 0 --policy-json "${POLICY_JSON}" --reference-json "${CORE_RING_RESULT}"
echo "Selective RGID-E0 audit complete. Meta-confirm/Val450/official instances/test were not accessed."
