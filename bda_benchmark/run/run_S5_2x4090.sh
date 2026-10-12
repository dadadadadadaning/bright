#!/usr/bin/env bash
set -euo pipefail

# S5_HCDM on TWO RTX 4090 GPUs with DistributedDataParallel.
# Clean ablation: Official DamageFormer + HCDM only.
# Global EBS = 1 sample/GPU x 2 GPUs x 8 accumulation = 16.

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
TRAIN_SCRIPT="script/standard_ML/train_S5_S8_GPU_2GPU.py"
DATASET_ROOT="${PROJECT_ROOT}/dataset"
TRAIN_LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt"
OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/S5/BRIGHT"
LOG_ROOT="/root/shared-nvme/BRIGHT-train-logs/S5"
SEED=3407
MASTER_PORT=29551
TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="S5_HCDM_2x4090_EBS16_50k_seed${SEED}_${TS}"
LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"

# Keep empty for the clean S5 ablation. Do NOT initialize from S1C.
PRETRAINED_MODEL_PATH=""

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--pretrained_model_path "${PRETRAINED_MODEL_PATH}")
fi

echo "[S5-2GPU] project      : ${PROJECT_ROOT}"
echo "[S5-2GPU] experiment   : ${EXP_NAME}"
echo "[S5-2GPU] output root  : ${OUTPUT_ROOT}"
echo "[S5-2GPU] log          : ${LOG_FILE}"
echo "[S5-2GPU] visible GPUs : 0,1"
echo "[S5-2GPU] DDP          : 2 processes / NCCL"
echo "[S5-2GPU] global EBS   : 1 x 2 x 8 = 16"

nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
      --experiment S5 \
      --train_dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --val_dataset_path "${DATASET_ROOT}" \
      --val_data_list_path "${VAL_LIST}" \
      --output_root "${OUTPUT_ROOT}" \
      --exp_name "${EXP_NAME}" \
      --image_size 640 \
      --train_batch_size 1 \
      --eval_batch_size 1 \
      --accumulation_steps 8 \
      --num_workers 4 \
      --max_steps 50000 \
      --learning_rate 1e-4 \
      --weight_decay 5e-3 \
      --seed "${SEED}" \
      --amp_dtype fp16 \
      --diff_channels 64 \
      --gamma_init 0.0 \
      --ce_weight 1.0 \
      --lovasz_weight 1.0 \
      --log_interval 20 \
      --val_interval 1000 \
      --val_max_batches 0 \
      --save_interval 5000 \
      "${EXTRA_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[S5-2GPU] started launcher PID=${PID}"
echo "[S5-2GPU] monitor log: tail -f ${LOG_FILE}"
echo "[S5-2GPU] monitor GPU: watch -n 2 nvidia-smi"
