#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
mkdir -p /root/shared-nvme/BRIGHT-checkpoints/S1C
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  script/standard_ML/train_DamageFormer_S1C_accum_ddp.py \
  --dataset BRIGHT \
  --train_batch_size_per_gpu 4 \
  --accumulation_steps 2 \
  --optimizer_steps 50000 \
  --max_iters 800000 \
  --eval_batch_size 1 \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --guidance_beta 1.0 \
  --eval_every 500 \
  --log_every 10 \
  --early_stop_patience 0 \
  --seed 3407 \
  --model_type S1C_preFusion_beta1_EBS16_50k \
  --model_param_path /root/shared-nvme/BRIGHT-checkpoints/S1C \
  --train_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --train_data_list_path dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --test_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
  --num_workers 8
