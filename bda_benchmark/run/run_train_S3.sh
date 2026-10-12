#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
mkdir -p /root/shared-nvme/BRIGHT-checkpoints/S3
CUDA_VISIBLE_DEVICES=0 python \
  script/standard_ML/train_DamageFormer_S3_BGMP_accum.py \
  --dataset BRIGHT \
  --train_batch_size 4 \
  --accumulation_steps 4 \
  --optimizer_steps 50000 \
  --max_iters 800000 \
  --eval_batch_size 1 \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --roi_alpha 0.1 \
  --proto_dim 128 \
  --proto_step_momentum 0.99 \
  --proto_tau 0.1 \
  --proto_warmup_iters 5000 \
  --eval_every 500 \
  --log_every 10 \
  --early_stop_patience 0 \
  --seed 3407 \
  --model_type S3_BGMP_K3_EBS16_50k \
  --model_param_path /root/shared-nvme/BRIGHT-checkpoints/S3 \
  --train_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --train_data_list_path dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --test_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
  --num_workers 8
