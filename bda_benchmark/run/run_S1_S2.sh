#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/PATH/TO/BRIGHT"
SAVE_ROOT="/PATH/TO/CHECKPOINTS"

# Run from BRIGHT-master/bda_benchmark

# S1: 2 x RTX 4090, local BS=4/GPU, ACC=2, effective BS=16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  script/standard_ML/train_DamageFormer_BG_accum_ddp.py \
  --dataset BRIGHT \
  --train_batch_size_per_gpu 4 \
  --accumulation_steps 2 \
  --optimizer_steps 50000 \
  --max_iters 800000 \
  --eval_batch_size 1 \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --roi_alpha 0.1 \
  --eval_every 500 \
  --log_every 10 \
  --seed 3407 \
  --model_type S1_BG_alpha01_EBS16_50k \
  --model_param_path "${SAVE_ROOT}" \
  --train_dataset_path "${DATA_ROOT}" \
  --train_data_list_path dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path "${DATA_ROOT}" \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --test_dataset_path "${DATA_ROOT}" \
  --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
  --num_workers 8

# S2 should be launched separately on the 5090 server:
# CUDA_VISIBLE_DEVICES=0 python \
#   script/standard_ML/train_DamageFormer_BGMP_accum.py \
#   --dataset BRIGHT \
#   --train_batch_size 4 \
#   --accumulation_steps 4 \
#   --optimizer_steps 50000 \
#   --max_iters 800000 \
#   --eval_batch_size 1 \
#   --crop_size 640 \
#   --learning_rate 1e-4 \
#   --weight_decay 5e-3 \
#   --roi_alpha 0.1 \
#   --proto_k 1 \
#   --proto_dim 128 \
#   --proto_step_momentum 0.99 \
#   --proto_tau 0.1 \
#   --proto_warmup_iters 5000 \
#   --eval_every 500 \
#   --log_every 10 \
#   --seed 3407 \
#   --model_type S2_BGMP_K1_EBS16_50k \
#   --model_param_path "${SAVE_ROOT}" \
#   --train_dataset_path "${DATA_ROOT}" \
#   --train_data_list_path dataset/splitname/standard_ML/train_set.txt \
#   --val_dataset_path "${DATA_ROOT}" \
#   --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
#   --test_dataset_path "${DATA_ROOT}" \
#   --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
#   --num_workers 8
