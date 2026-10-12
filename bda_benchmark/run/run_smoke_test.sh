#!/usr/bin/env bash
set -euo pipefail
EXP=${1:-S5}
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
export DATA_ROOT=/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset
export TRAIN_LIST=${DATA_ROOT}/splitname/standard_ML/train_set.txt
export VAL_LIST=${DATA_ROOT}/splitname/standard_ML/val_set.txt

CUDA_VISIBLE_DEVICES=0 python -u \
  script/standard_ML/train_Hierarchical_RGBSAR.py \
  --experiment "$EXP" \
  --train_dataset_path "$DATA_ROOT" \
  --train_data_list_path "$TRAIN_LIST" \
  --val_dataset_path "$DATA_ROOT" \
  --val_data_list_path "$VAL_LIST" \
  --output_root /root/shared-nvme/BRIGHT-checkpoints/${EXP}/debug \
  --exp_name ${EXP}_smoke_test \
  --image_size 640 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --accumulation_steps 2 \
  --optimizer_steps 20 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --amp_dtype bf16 \
  --diff_channels 64 \
  --gamma_init 0.0 \
  --roi_alpha 0.1 \
  --align_dim 128 \
  --lambda_ia 0.005 \
  --ia_warmup 8 \
  --ia_ramp 4 \
  --seed 3407 \
  --num_workers 0 \
  --log_interval 1 \
  --val_interval 10 \
  --val_max_batches 4 \
  --save_interval 10
