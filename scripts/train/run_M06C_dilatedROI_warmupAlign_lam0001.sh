#!/bin/bash
set -e

cd /root/shared-nvme/BRIGHT-master

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py \
  --dataset BRIGHT \
  --train_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --train_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/val_set.txt \
  --model_param_path /root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final/M06C_dilatedROI_warmupAlign_lam0001 \
  --crop_size 512 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 2 \
  --max_iters 50000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --use_lovasz \
  --roi_mode dilated_soft \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --detach_mask \
  --use_align \
  --align_type damage_aware \
  --lambda_align 0.0001 \
  --align_warmup_iters 10000 \
  --align_ramp_iters 10000
