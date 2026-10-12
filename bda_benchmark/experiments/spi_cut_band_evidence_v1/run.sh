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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_cut_band_evidence_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"
echo "Cut-band Evidence Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" \
    "${SCRIPT_DIR}/audit_e0.py" "${SCRIPT_DIR}/train_verifier.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -c "import scipy,torch,imageio; print('dependencies:', scipy.__version__, torch.__version__, imageio.__version__)"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Cut-band Evidence preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
RGID_DIR="${RGID_DIR:-/root/data/BRIGHT-checkpoints/SPI_STRUCTURED_PLUGINS_V1/strip_rgid_step500_seed${SEED}}"
RGID_CKPT="${RGID_CKPT:-${RGID_DIR}/rgid_step_500_calibrated.pth}"
RGID_PROVENANCE="${RGID_PROVENANCE:-${RGID_DIR}/provenance.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CUT_BAND_EVIDENCE_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CUT_BAND_EVIDENCE_V1}"
TRAIN_CACHE="${OUTPUT_ROOT}/meta_train.pt"
CAL_CACHE="${OUTPUT_ROOT}/meta_calibration.pt"
ADMISSION_JSON="${RESULT_ROOT}/cut_band_e0_seed${SEED}.json"
PREREQUISITE_JSON="${PREREQUISITE_JSON:-}"

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" "${RGID_CKPT}" \
  "${RGID_PROVENANCE}" "${SPLIT_ROOT}/split_manifest.json" "${SPLIT_ROOT}/meta_train.txt" \
  "${SPLIT_ROOT}/meta_calibration.txt"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
TRAIN_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$(${PYTHON_BIN} -c "import json; print(json.load(open('${SPLIT_ROOT}/split_manifest.json'))['counts']['meta_calibration'])")"

build_cache() {
  local split="$1"; local list_path="$2"; local expected="$3"; local output="$4"; local maximum="$5"
  local profile_args=()
  [[ "${PROFILE_TIMING:-0}" == "1" ]] && profile_args+=(--profile-timing)
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cut_band_evidence_v1.build_cache \
    --checkpoint "${P3_CKPT}" --rgid-checkpoint "${RGID_CKPT}" \
    --rgid-provenance "${RGID_PROVENANCE}" --rgid-train-list "${SPLIT_ROOT}/meta_train.txt" \
    --dataset-path "${DATASET_ROOT}" --list-path "${list_path}" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${output}" --split "${split}" --expected-images "${expected}" --max-images "${maximum}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --minimum-area "${MINIMUM_AREA:-32}" --band-radius "${BAND_RADIUS:-2}" \
    --matching-max-side "${MATCHING_MAX_SIDE:-64}" --matching-window "${MATCHING_WINDOW:-5}" \
    --matching-channels "${MATCHING_CHANNELS:-32}" \
    --state-every "${STATE_EVERY:-25}" --seed "${SEED}" "${profile_args[@]}"
}

if [[ "${MODE}" == "smoke" ]]; then
  build_cache meta_train "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${OUTPUT_ROOT}/smoke.pt" "${SMOKE_IMAGES:-20}"
  echo "Cut-band Evidence real-data smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
  exit 0
fi

if [[ "${MODE}" == "audit" || "${MODE}" == "all" ]]; then
  test -n "${PREREQUISITE_JSON}" || {
    echo "Set PREREQUISITE_JSON to the completed Selective Strip-RGID E0 result." >&2
    exit 1
  }
  test -s "${PREREQUISITE_JSON}" || { echo "Missing prerequisite: ${PREREQUISITE_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${PREREQUISITE_JSON}')); g=p.get('gains_miou_pp',{}).get('r4_minus_r0',-999); assert g >= 0.20, 'R4 split-oracle ceiling is too low for cut-band selection'; assert p.get('decision') == 'STOP_SELECTIVE_RGID_NO_DEPLOYABLE_GAIN', 'Cut-band E0 is only for R4-high/R2-failed branch'; print('Prerequisite passed: R4 high, existing selector failed, gain=',g)"
  [[ -s "${TRAIN_CACHE}" ]] || build_cache meta_train "${SPLIT_ROOT}/meta_train.txt" "${TRAIN_COUNT}" "${TRAIN_CACHE}" 0
  [[ -s "${CAL_CACHE}" ]] || build_cache meta_calibration "${SPLIT_ROOT}/meta_calibration.txt" "${CAL_COUNT}" "${CAL_CACHE}" 0
  if [[ -s "${ADMISSION_JSON}" ]]; then
    echo "Existing E0 result found; refusing to reuse meta_calibration interactively."
  else
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cut_band_evidence_v1.audit_e0 \
      --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" --output "${ADMISSION_JSON}" \
      --utility-margin "${UTILITY_MARGIN:-0.05}" --permutations "${PERMUTATIONS:-1000}" --seed "${SEED}"
  fi
  [[ "${MODE}" == "audit" ]] && exit 0
fi

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  test -s "${TRAIN_CACHE}" || { echo "Run audit first: missing ${TRAIN_CACHE}" >&2; exit 1; }
  test -s "${CAL_CACHE}" || { echo "Run audit first: missing ${CAL_CACHE}" >&2; exit 1; }
  test -s "${ADMISSION_JSON}" || { echo "Run audit first: missing ${ADMISSION_JSON}" >&2; exit 1; }
  test ! -e "${OUTPUT_ROOT}/verifier_e1/calibration_result.json" || {
    echo "Frozen E1 calibration result already exists; refusing a repeated calibration evaluation." >&2
    exit 1
  }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cut_band_evidence_v1.train_verifier \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --admission-json "${ADMISSION_JSON}" --output-dir "${OUTPUT_ROOT}/verifier_e1" \
    --groups "${VERIFIER_GROUPS:-geometry,rgb_edge,sar_edge,same_position,local_matching}" \
    --steps "${VERIFIER_STEPS:-1000}" --hidden "${VERIFIER_HIDDEN:-32}" --seed "${SEED}"
fi
