#!/bin/bash
set -e
set -o pipefail

cd /root/shared-nvme/BRIGHT-master

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bright

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="/root/shared-nvme/BRIGHT-master/logs/BGCA_rebuild"
WEIGHT_DIR="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_rebuild/M06D_dilatedROI_classAwareAlign_lam0001"
LOG_FILE="${LOG_DIR}/M06D_dilatedROI_classAwareAlign_lam0001.log"

mkdir -p "${LOG_DIR}"
mkdir -p "${WEIGHT_DIR}"

python bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py \
  --dataset BRIGHT \
  --train_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --train_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/train_set.txt \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/val_set.txt \
  --model_param_path "${WEIGHT_DIR}" \
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
  --align_type class_aware \
  --lambda_align 0.0001 \
  --align_margin 0.3 \
  --align_pull_weight 1.0 \
  --align_margin_weight 0.2 \
  --align_warmup_iters 10000 \
  --align_ramp_iters 10000 \
  2>&1 | tee "${LOG_FILE}"
