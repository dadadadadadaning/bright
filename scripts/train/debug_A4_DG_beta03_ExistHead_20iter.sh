#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master
export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py \
  --dataset DEBUG_A4_DG_beta03_ExistHead \
  --train_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --train_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/val_set.txt \
  --crop_size 640 --train_batch_size 1 --eval_batch_size 1 --num_workers 0 --max_iters 20 \
  --learning_rate 1e-4 --weight_decay 5e-3 \
  --model_param_path /root/shared-nvme/BRIGHT-master/saved_weights/debug \
  --log_interval 1 --val_interval 20 --val_max_batches 1 \
  --roi_mode dilated_soft --roi_fusion multiplicative --roi_beta 1.0 --roi_alpha 0.1 --roi_dilate 5 \
  --lambda_align 0.0 \
  --use_mddm --mddm_variant dual_scale_gate --mddm_gamma_init 0.0 --mddm_use_roi \
  --dg_gate_reduction 4 --dg_gamma_init 0.0 --dg_gamma_mode sigmoid01 --dg_large_beta 0.3 \
  --use_damage_exist_head --lambda_exist 0.1 \
  --use_lovasz --ce_weight 1.0 --lovasz_weight 1.0 --save_policy smart
