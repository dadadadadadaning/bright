#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|pilot|calibration|full|audit|auto) ;;
  *) echo "Choose preflight|smoke|pilot|calibration|full|audit|auto" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
cd "${BDA_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n "${SCRIPT_DIR}/run.sh"
  python -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/audit.py"
  python -m unittest discover -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
  echo "SPI S2 Distillation V1 isolated preflight passed"
  exit 0
fi

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
CALIBRATION_LIST="${CALIBRATION_LIST:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_RISK_GATE_V1/seed3407/split_seed3407/gate_calibration.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SEED="${SEED:-3407}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_S2_DISTILLATION_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_S2_DISTILLATION_V1}"
EXPECTED_SHA256="${EXPECTED_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

for required in "${DATASET_ROOT}" "${TRAIN_LIST}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
[[ -z "${EXPECTED_SHA256}" || "${observed}" == "${EXPECTED_SHA256}" ]] || {
  echo "P3 identity mismatch: ${observed}" >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"

train_model() {
  local steps="$1"
  local output_root="$2"
  python -u -m bda_benchmark.experiments.spi_s2_distillation_v1.train \
    --dataset-path "${DATASET_ROOT}" --train-list "${TRAIN_LIST}" \
    --base-checkpoint "${P3_CKPT}" --output-dir "${output_root}" \
    --seed "${SEED}" --device cuda:0 --crop-size "${CROP_SIZE:-640}" \
    --batch-size "${BATCH_SIZE:-4}" --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
    --optimizer-steps "${steps}" --num-workers "${NUM_WORKERS:-4}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" --weight-decay "${WEIGHT_DECAY:-5e-3}" \
    --hidden-channels "${HIDDEN_CHANNELS:-64}" --lambda-s2 "${LAMBDA_S2:-0.50}" \
    --lambda-anchor "${LAMBDA_ANCHOR:-0.10}" --log-every "${LOG_EVERY:-20}" \
    --save-every "${SAVE_EVERY:-100}" ${PROFILE_TIMING:+--profile-timing}
}

audit_model() {
  local steps="$1"
  local list="$2"
  local expected_images="$3"
  local max_images="$4"
  local expected_p3="$5"
  local output_json="$6"
  local checkpoint_root="$7"
  python -u -m bda_benchmark.experiments.spi_s2_distillation_v1.audit \
    --base-checkpoint "${P3_CKPT}" --student-checkpoint "${checkpoint_root}/joint_last.pth" \
    --dataset-path "${DATASET_ROOT}" --val-list "${list}" --sar-stats-json "${SAR_STATS_JSON}" \
    --output-json "${output_json}" --device cuda:0 --num-workers "${NUM_WORKERS:-4}" \
    --expected-images "${expected_images}" --expected-steps "${steps}" --max-images "${max_images}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}" \
    --expected-p3-miou "${expected_p3}" --progress-every "${PROGRESS_EVERY:-10}"
}

if [[ "${MODE}" == "smoke" ]]; then
  smoke_steps="${SMOKE_STEPS:-2}"
  smoke_images="${SMOKE_IMAGES:-2}"
  smoke_root="${CHECKPOINT_ROOT}/smoke${smoke_steps}"
  test -s "${CALIBRATION_LIST}" || { echo "Missing calibration list: ${CALIBRATION_LIST}" >&2; exit 1; }
  train_model "${smoke_steps}" "${smoke_root}"
  audit_model "${smoke_steps}" "${CALIBRATION_LIST}" 291 "${smoke_images}" -1 \
    "${RESULT_ROOT}/SPI_S2_DISTILLATION_smoke${smoke_images}_seed${SEED}.json" "${smoke_root}"
  exit 0
fi

if [[ "${MODE}" == "pilot" ]]; then
  train_model "${PILOT_STEPS:-500}" "${CHECKPOINT_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "calibration" ]]; then
  test -s "${CALIBRATION_LIST}" || { echo "Missing calibration list: ${CALIBRATION_LIST}" >&2; exit 1; }
  steps="${EXPECTED_STEPS:-${PILOT_STEPS:-500}}"
  audit_model "${steps}" "${CALIBRATION_LIST}" "${CALIBRATION_IMAGES:-291}" 0 -1 \
    "${RESULT_ROOT}/SPI_S2_DISTILLATION_internal_calibration_step${steps}_seed${SEED}.json" "${CHECKPOINT_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "full" ]]; then
  train_model "${FULL_STEPS:-3000}" "${CHECKPOINT_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "audit" ]]; then
  steps="${EXPECTED_STEPS:-${FULL_STEPS:-3000}}"
  audit_model "${steps}" "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}" 0 "${EXPECTED_P3_MIOU:-0.605020}" \
    "${RESULT_ROOT}/SPI_S2_DISTILLATION_validation_full_step${steps}_seed${SEED}.json" "${CHECKPOINT_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "auto" ]]; then
  pilot_steps="${PILOT_STEPS:-500}"
  full_steps="${FULL_STEPS:-3000}"
  train_model "${pilot_steps}" "${CHECKPOINT_ROOT}"
  pilot_json="${RESULT_ROOT}/SPI_S2_DISTILLATION_internal_calibration_step${pilot_steps}_seed${SEED}.json"
  test -s "${CALIBRATION_LIST}" || { echo "Missing calibration list: ${CALIBRATION_LIST}" >&2; exit 1; }
  audit_model "${pilot_steps}" "${CALIBRATION_LIST}" "${CALIBRATION_IMAGES:-291}" 0 -1 \
    "${pilot_json}" "${CHECKPOINT_ROOT}"
  pilot_decision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "${pilot_json}")"
  if [[ "${pilot_decision}" != "PROMOTE_S2_DISTILLATION_TO_FULL" ]]; then
    echo "Automatic stop after pilot: ${pilot_decision}"
    echo "Val450 was not accessed."
    exit 0
  fi
  train_model "${full_steps}" "${CHECKPOINT_ROOT}"
  full_calibration_json="${RESULT_ROOT}/SPI_S2_DISTILLATION_internal_calibration_step${full_steps}_seed${SEED}.json"
  audit_model "${full_steps}" "${CALIBRATION_LIST}" "${CALIBRATION_IMAGES:-291}" 0 -1 \
    "${full_calibration_json}" "${CHECKPOINT_ROOT}"
  full_decision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "${full_calibration_json}")"
  case "${full_decision}" in
    S2_DISTILLED_MODEL_SURPASSES_FIXED_S2|S2_DISTILLED_MODEL_VALID_BELOW_FIXED_S2)
      audit_model "${full_steps}" "${VAL_LIST}" "${EXPECTED_VAL_IMAGES:-450}" 0 "${EXPECTED_P3_MIOU:-0.605020}" \
        "${RESULT_ROOT}/SPI_S2_DISTILLATION_validation_full_step${full_steps}_seed${SEED}.json" "${CHECKPOINT_ROOT}" ;;
    *) echo "Automatic stop after full internal calibration: ${full_decision}"; echo "Val450 was not accessed." ;;
  esac
fi
