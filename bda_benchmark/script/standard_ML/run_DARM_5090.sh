#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
CUDA_VISIBLE_DEVICES=0 python -u script/standard_ML/train_DARM.py \
  --train_dataset_path dataset \
  --train_data_list_path dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path dataset \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --output_root /root/shared-nvme/BRIGHT-checkpoints/DARM/BRIGHT \
  --exp_name DARM_P5_HSD \
  --image_size 640 --train_batch_size 2 --eval_batch_size 1 \
  --accumulation_steps 8 --max_steps 50000 --learning_rate 1e-4 --weight_decay 5e-3 \
  --amp_dtype bf16 --topk_ratio 0.10 --boundary_radius 1 --ring_radius 3 \
  --lambda_inst 0.2 --inst_warmup 3000 --inst_ramp 3000 \
  --lambda_ia 0.005 --ia_warmup 8000 --ia_ramp 2000
