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
SAVE_ROOT="$ROOT/saved_weights/BGCA_priority_E1_E5"
LOG_ROOT="$ROOT/logs/train_priority_E1_E5"
EXP_NAME="E2_BGCA_HDC_seed3407_C640_I120k_BS2"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

echo "[INFO] E2 = BGCA + HDC"
echo "[INFO] ROOT=$ROOT"
echo "[INFO] GPU_ID=$GPU_ID"
echo "[INFO] EXP_NAME=$EXP_NAME"

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$TRAIN_PY" \
  --dataset BRIGHT \
  --exp_name "$EXP_NAME" \
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
  --roi_mode none \
  --use_hdc \
  --hdc_mode blend \
  --hdc_alpha 0.3 \
  --lambda_hdc 0.5 \
  --use_damage_exist_head \
  --lambda_exist 0.1 \
  --use_severity_head \
  --lambda_severity 0.1 \
  --save_policy smart \
  2>&1 | tee "$LOG_ROOT/${EXP_NAME}.log"
