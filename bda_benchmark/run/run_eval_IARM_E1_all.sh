#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT_DIR="${CKPT_DIR:-/root/shared-nvme/BRIGHT-checkpoints/IARM/BRIGHT/IARM_E1_CMU_NORMAL_Stage4_2x4090_EBS16_50k_E1_CMU_NORMAL_20260714_223753}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-none}"

cd "${PROJECT_ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_JSON="${CKPT_DIR}/config.json"

CANDIDATES=(
  "best_val_miou.pth"
  "best_val_damage_hf1.pth"
  "best_val_event_macro_target3.pth"
)

FOUND=0

for CHECKPOINT_NAME in "${CANDIDATES[@]}"; do
  CHECKPOINT="${CKPT_DIR}/${CHECKPOINT_NAME}"

  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "SKIP missing checkpoint: ${CHECKPOINT}"
    continue
  fi

  FOUND=1
  STEM="${CHECKPOINT_NAME%.pth}"
  OUTPUT_JSON="${CKPT_DIR}/eval_E1_${STEM}_extended.json"
  LOG_FILE="${CKPT_DIR}/eval_E1_${STEM}_extended.log"

  echo "============================================================"
  echo "Evaluating checkpoint : ${CHECKPOINT}"
  echo "Config                : ${CONFIG_JSON}"
  echo "Output                : ${OUTPUT_JSON}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python \
    script/standard_ML/eval_IARM_E1_diagnostics.py \
    --checkpoint "${CHECKPOINT}" \
    --config_json "${CONFIG_JSON}" \
    --dataset_path "${PROJECT_ROOT}/dataset" \
    --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
    --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
    --split both \
    --eval_batch_size 1 \
    --num_workers "${NUM_WORKERS}" \
    --device cuda:0 \
    --amp "${AMP}" \
    --diagnostic_samples_per_image_class 48 \
    --max_diagnostic_samples_per_class 30000 \
    --output_json "${OUTPUT_JSON}" \
    2>&1 | tee "${LOG_FILE}"
done

if [[ "${FOUND}" -eq 0 ]]; then
  echo "ERROR: no expected best checkpoint was found under ${CKPT_DIR}"
  exit 3
fi
