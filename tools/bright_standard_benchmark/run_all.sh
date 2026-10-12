#!/usr/bin/env bash
set -euo pipefail

# Unified six-model BRIGHT evaluation. Override any path through environment
# variables when the server layout differs.
PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/bda_benchmark/dataset}"
LIST_FILE="${LIST_FILE:-${DATA_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/STANDARD_MODELS_UNIFIED_V1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TTA="${TTA:-0}"
AMP="${AMP:-0}"
FORCE="${FORCE:-0}"

EVALUATOR="${PROJECT_ROOT}/tools/bright_standard_benchmark/evaluate_model.py"
REPORTER="${PROJECT_ROOT}/tools/bright_standard_benchmark/build_report.py"
MAMBA_CFG="${PROJECT_ROOT}/ChangeMamba-master/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"

declare -A CHECKPOINTS=(
  [unet]="${CHECKPOINT_ROOT}/ckpt_UNet_standard_ML_split.pth"
  [siamcrnn]="${CHECKPOINT_ROOT}/ckpt_SiamCRNN_standard_ML_split.pth"
  [siamattnunet]="${CHECKPOINT_ROOT}/ckpt_SiamAttnUNet_standard_ML_split.pth"
  [deeplabv3plus]="${CHECKPOINT_ROOT}/ckpt_DeepLabV3Plus_standard_ML_split.pth"
  [damageformer]="${CHECKPOINT_ROOT}/ckpt_DamageFormer_standard_ML_split.pth"
  [changemamba]="${CHECKPOINT_ROOT}/ckpt_ChangeMamba_bright_standard_ML_split.pth"
)

MODELS=(unet siamcrnn siamattnunet deeplabv3plus damageformer changemamba)

for required in "${EVALUATOR}" "${REPORTER}" "${DATA_ROOT}" "${LIST_FILE}" "${MAMBA_CFG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Missing required path: ${required}" >&2
    exit 2
  fi
done
for model in "${MODELS[@]}"; do
  if [[ ! -f "${CHECKPOINTS[$model]}" ]]; then
    echo "[ERROR] Missing checkpoint for ${model}: ${CHECKPOINTS[$model]}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

COMMON_ARGS=(
  --dataset-root "${DATA_ROOT}"
  --list-file "${LIST_FILE}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --batch-size "${BATCH_SIZE}"
)
if [[ "${TTA}" == "1" ]]; then
  COMMON_ARGS+=(--tta)
fi
if [[ "${AMP}" == "1" ]]; then
  COMMON_ARGS+=(--amp)
fi

for model in "${MODELS[@]}"; do
  if [[ "${FORCE}" != "1" && -f "${OUTPUT_ROOT}/${model}/metrics.json" ]]; then
    echo "[SKIP] ${model}: metrics.json already exists (set FORCE=1 to rerun)"
    continue
  fi
  echo "[RUN] ${model}"
  EXTRA_ARGS=()
  if [[ "${model}" == "changemamba" ]]; then
    EXTRA_ARGS+=(--changemamba-config "${MAMBA_CFG}")
  fi
  "${PYTHON_BIN}" "${EVALUATOR}" \
    --model "${model}" \
    --checkpoint "${CHECKPOINTS[$model]}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/${model}.log"
done

"${PYTHON_BIN}" "${REPORTER}" \
  --results-root "${OUTPUT_ROOT}" \
  --dataset-root "${DATA_ROOT}"

echo "[DONE] Results and figures: ${OUTPUT_ROOT}"
