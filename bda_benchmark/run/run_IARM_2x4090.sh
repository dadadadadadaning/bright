#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/IARM"
LOG_ROOT="/root/shared-nvme/BRIGHT-train-logs/IARM"
EXPERIMENT="${EXPERIMENT:-E2_IAR_EXIST}"
RESIDUAL_STAGES="${RESIDUAL_STAGES:-4}"
MASTER_PORT="${MASTER_PORT:-29591}"
TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="IARM_${EXPERIMENT}_Stage${RESIDUAL_STAGES//,/}_2x4090_EBS16_50k"
LOG_FILE="${LOG_ROOT}/${MODEL_TAG}_${TS}.log"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  script/standard_ML/train_DamageFormer_IARM_accum_ddp.py \
  --dataset BRIGHT \
  --train_dataset_path "${PROJECT_ROOT}/dataset" \
  --train_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt" \
  --val_dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_dataset_path "${PROJECT_ROOT}/dataset" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --experiment "${EXPERIMENT}" \
  --residual_stages "${RESIDUAL_STAGES}" \
  --residual_dim 128 \
  --flow_hidden_dim 64 \
  --max_offset_pixels 2.0 \
  --detach_backbone_for_residual \
  --residual_gamma_max 0.25 \
  --roi_alpha 0.1 \
  --gaussian_step_momentum 0.99 \
  --gaussian_variance_floor 0.002 \
  --max_pixels_per_image_class 256 \
  --cmu_temperature 0.1 \
  --cmu_max_samples 256 \
  --contrast_temperature 0.1 \
  --contrast_samples_per_image_class 12 \
  --train_batch_size_per_gpu 4 \
  --eval_batch_size 1 \
  --accumulation_steps 2 \
  --optimizer_steps 50000 \
  --max_iters 800000 \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --eval_every 500 \
  --log_every 10 \
  --num_workers 4 \
  --seed 3407 \
  --model_type "${MODEL_TAG}" \
  --model_param_path "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_FILE}"
