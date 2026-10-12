#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${1:-${GPU_ID:-0}}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

TRAIN_PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py"
DATA="$ROOT/bda_benchmark/dataset"
TRAIN_LIST="$DATA/splitname/standard_ML/train_set.txt"
VAL_LIST="$DATA/splitname/standard_ML/val_set.txt"
SAVE_ROOT="$ROOT/saved_weights/BGCA_final"
LOG_ROOT="$ROOT/logs/train_SFDG"
EXP_NAME="A11_Cons005_w20000_SH_Under005_SFDG_seed3407_C640_I120k_BS2"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$TRAIN_PY" \
  --dataset "$EXP_NAME" \
  --train_dataset_path "$DATA" \
  --train_data_list_path "$TRAIN_LIST" \
  --val_dataset_path "$DATA" \
  --val_data_list_path "$VAL_LIST" \
  --model_param_path "$SAVE_ROOT" \
  --crop_size 640 \
  --train_batch_size 2 \
  --eval_batch_size 1 \
  --num_workers 4 \
  --max_iters 120000 \
  --seed 3407 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
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
  --exist_loss_type bce \
  --exist_focal_gamma 2.0 \
  --use_exist_consistency \
  --lambda_exist_cons 0.05 \
  --exist_cons_warmup 20000 \
  --exist_cons_mode mse_detach \
  --use_severity_head \
  --lambda_severity 0.1 \
  --severity_warmup 20000 \
  --use_damage_under_loss \
  --lambda_under 0.005 \
  --under_warmup 20000 \
  --under_margin 0.05 \
  --use_sfdg \
  --sfdg_gamma 0.5 \
  --save_policy smart \
  2>&1 | tee "$LOG_ROOT/${EXP_NAME}.log"
