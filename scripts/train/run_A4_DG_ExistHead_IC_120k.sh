#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master
export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

TRAIN_SCRIPT="/root/shared-nvme/BRIGHT-master/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py"
DATASET_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
TRAIN_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
SAVE_ROOT="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final"
LOG_DIR="/root/shared-nvme/BRIGHT-master/logs/BGCA_aux"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" nohup python "${TRAIN_SCRIPT}" \
  --dataset A4_DG_ExistHead_IC_dilatedROI_crop640_iter120k_bs2 \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --crop_size 640 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 2 \
  --max_iters 120000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --model_param_path "${SAVE_ROOT}" \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --lambda_align 0.0 \
  --use_mddm \
  --mddm_variant dual_scale_gate \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --dg_gate_reduction 4 \
  --dg_gamma_init 0.0 \
  --dg_gamma_mode sigmoid01 \
  --use_damage_exist_head \
  --lambda_exist 0.1 \
  --use_intact_consistency \
  --lambda_ic 0.01 \
  --ic_dim 256 \
  --ic_normalize \
  --use_lovasz \
  --ce_weight 1.0 \
  --lovasz_weight 1.0 \
  --save_policy smart \
  > "${LOG_DIR}/A4_DG_ExistHead_IC_120k.log" 2>&1 &

echo "Started A4_DG_ExistHead_IC. Log: ${LOG_DIR}/A4_DG_ExistHead_IC_120k.log"
