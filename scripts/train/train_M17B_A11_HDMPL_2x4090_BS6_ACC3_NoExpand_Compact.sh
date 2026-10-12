#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/BRIGHT-master"
GPU_IDS="${1:-0,1}"
MAX_ITERS="${2:-16000}"
LR="${3:-1e-4}"
EXP_NAME="M17B_A11_HDMPL_2x4090_BS6_ACC3_NoExpand_Compact_C640_I${MAX_ITERS}_LR${LR}_Seed3407"
SAVE_ROOT="$ROOT/saved_weights/M17"
LOG_ROOT="$ROOT/logs/M17"
TRAIN_PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06_dp_init.py"
DATA_ROOT="$ROOT/bda_benchmark/dataset"

cd "$ROOT"
mkdir -p "$SAVE_ROOT" "$SAVE_ROOT/$EXP_NAME" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python "$TRAIN_PY" \
  --dataset BRIGHT \
  --seed 3407 \
  --train_dataset_path "$DATA_ROOT" \
  --train_data_list_path "$DATA_ROOT/splitname/standard_ML/train_set.txt" \
  --val_dataset_path "$DATA_ROOT" \
  --val_data_list_path "$DATA_ROOT/splitname/standard_ML/val_set.txt" \
  --model_param_path "$SAVE_ROOT" \
  --exp_name "$EXP_NAME" \
  --device auto \
  --crop_size 640 \
  --train_batch_size 6 \
  --eval_batch_size 1 \
  --num_workers 4 \
  --max_iters "$MAX_ITERS" \
  --accumulation_steps 3 \
  --learning_rate "$LR" \
  --weight_decay 5e-3 \
  --log_interval 200 \
  --val_interval 2000 \
  --save_interval 2000 \
  --save_policy smart \
  --use_lovasz \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --use_damage_exist_head \
  --lambda_exist 0.1 \
  --exist_pos_weight 2.0 \
  --exist_loss_type bce \
  --use_hdmpl \
  --lambda_hdmpl 0.005 \
  --hdmpl_warmup 8000 \
  --hdmpl_ramp 2000 \
  --hdmpl_proto_dim 128 \
  --hdmpl_max_tokens_per_class 256 \
  --hdmpl_hard_topk_ratio 0.5 \
  --hdmpl_temperature 0.1 \
  --hdmpl_momentum 0.99 \
  --hdmpl_queue_size 2048 \
  --hdmpl_init_damaged_tokens 768 \
  2>&1 | tee "$LOG_ROOT/${EXP_NAME}.log"
