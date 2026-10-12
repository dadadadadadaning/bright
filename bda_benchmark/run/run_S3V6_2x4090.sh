#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/S3V6"
LOG_ROOT="/root/shared-nvme/BRIGHT-train-logs/S3V6"
MASTER_PORT="${MASTER_PORT:-29583}"
EVENT_LOSS_ALPHA="${EVENT_LOSS_ALPHA:-0.0}"
TS="$(date +%Y%m%d_%H%M%S)"

if [[ "${EVENT_LOSS_ALPHA}" == "0" || "${EVENT_LOSS_ALPHA}" == "0.0" ]]; then
  VARIANT="A_MainProtected_EventBalancedProto"
else
  VARIANT="B_MainProtected_EventBalancedProto_EventLoss${EVENT_LOSS_ALPHA}"
fi

EXP_NAME="S3V6_${VARIANT}_2x4090_EBS16_50k_${TS}"
LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  script/standard_ML/train_DamageFormer_BGMP_K3EventBalanced_accum_ddp.py \
  --dataset BRIGHT \
  --train_dataset_path "${PROJECT_ROOT}/dataset" \
  --train_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt" \
  --val_dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_dataset_path "${PROJECT_ROOT}/dataset" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --train_batch_size_per_gpu 4 \
  --eval_batch_size 1 \
  --accumulation_steps 2 \
  --optimizer_steps 50000 \
  --max_iters 800000 \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --roi_alpha 0.1 \
  --proto_dim 256 \
  --proto_step_momentum 0.99 \
  --proto_tau 0.1 \
  --proto_warmup_iters 5000 \
  --core_confidence 0.70 \
  --lambda_damaged 0.05 \
  --lambda_destroyed 0.05 \
  --proto_evidence_clip 4.0 \
  --proto_evidence_margin 0.03 \
  --max_pixels_per_image_role 2048 \
  --event_loss_alpha "${EVENT_LOSS_ALPHA}" \
  --event_weight_power 0.5 \
  --event_weight_min 0.5 \
  --event_weight_max 2.0 \
  --eval_every 500 \
  --log_every 10 \
  --seed 3407 \
  --model_type "${EXP_NAME}" \
  --model_param_path "${OUTPUT_ROOT}" \
  --num_workers 4 \
  2>&1 | tee "${LOG_FILE}"
