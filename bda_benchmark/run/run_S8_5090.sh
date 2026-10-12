#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master/bda_benchmark

export DATA_ROOT=/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset
export TRAIN_LIST=${DATA_ROOT}/splitname/standard_ML/train_set.txt
export VAL_LIST=${DATA_ROOT}/splitname/standard_ML/val_set.txt

CUDA_VISIBLE_DEVICES=0 python -u \
  script/standard_ML/train_Hierarchical_RGBSAR.py \
  --experiment S8 \
  --train_dataset_path "${DATA_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATA_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --output_root /root/shared-nvme/BRIGHT-checkpoints/S8/BRIGHT \
  --exp_name S8_Hierarchical_RGBSAR_EBS16_50k \
  --image_size 640 \
  --train_batch_size 2 \
  --eval_batch_size 1 \
  --accumulation_steps 8 \
  --optimizer_steps 50000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --amp_dtype bf16 \
  --diff_channels 64 \
  --gamma_init 0.0 \
  --roi_alpha 0.1 \
  --align_dim 128 \
  --lambda_ia 0.005 \
  --ia_warmup 8000 \
  --ia_ramp 2000 \
  --seed 3407 \
  --log_interval 20 \
  --val_interval 500 \
  --save_interval 5000
