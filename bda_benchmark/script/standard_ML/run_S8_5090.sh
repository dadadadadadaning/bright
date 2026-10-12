#!/usr/bin/env bash
set -euo pipefail

# S8_IA_MLA on a single RTX 5090.
# Cumulative S8: BG-AS-HCDM + intact-only Stage3/Stage4 alignment.
# EBS = 2 micro-batch x 8 accumulation x 1 GPU = 16.

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
TRAIN_SCRIPT="script/standard_ML/train_S5_S8_GPU.py"
DATASET_ROOT="${PROJECT_ROOT}/dataset"
TRAIN_LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt"
OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/S8/BRIGHT"
LOG_ROOT="/root/shared-nvme/BRIGHT-train-logs/S8"
SEED=3407
TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="S8_IA_MLA_EBS16_50k_seed${SEED}_${TS}"
LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"

# IMPORTANT: keep empty for the clean S8 cumulative ablation unless you have
# the exact S0 official DamageFormer baseline checkpoint under the same protocol.
# Do NOT initialize S8 from the S1C checkpoint.
PRETRAINED_MODEL_PATH=""

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--pretrained_model_path "${PRETRAINED_MODEL_PATH}")
fi

echo "[S8] project      : ${PROJECT_ROOT}"
echo "[S8] experiment   : ${EXP_NAME}"
echo "[S8] output root  : ${OUTPUT_ROOT}"
echo "[S8] log          : ${LOG_FILE}"
echo "[S8] GPU logical  : 0"
echo "[S8] EBS          : 2 x 8 x 1 = 16"

nohup env CUDA_VISIBLE_DEVICES=0 \
  python -u "${TRAIN_SCRIPT}" \
    --experiment S8 \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --output_root "${OUTPUT_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --image_size 640 \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --accumulation_steps 8 \
    --num_workers 4 \
    --max_steps 50000 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --seed "${SEED}" \
    --amp_dtype bf16 \
    --diff_channels 64 \
    --gamma_init 0.0 \
    --roi_alpha 0.1 \
    --align_dim 128 \
    --lambda_ia 0.005 \
    --ia_warmup 8000 \
    --ia_ramp 2000 \
    --ce_weight 1.0 \
    --lovasz_weight 1.0 \
    --log_interval 20 \
    --val_interval 1000 \
    --val_max_batches 0 \
    --save_interval 5000 \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[S8] started PID=${PID}"
echo "[S8] monitor: tail -f ${LOG_FILE}"
echo "[S8] GPU:     watch -n 2 nvidia-smi"
