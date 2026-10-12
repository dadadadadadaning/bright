#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${GPU_ID:-0}"

TRAIN_PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py"
DATA="$ROOT/bda_benchmark/dataset"
TRAIN_LIST="$DATA/splitname/standard_ML/train_set.txt"
VAL_LIST="$DATA/splitname/standard_ML/val_set.txt"

SAVE_ROOT="$ROOT/saved_weights/BGCA_clean/A11_next"
LOG_ROOT="$SAVE_ROOT/logs"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --train_dataset_path "$DATA"
  --train_data_list_path "$TRAIN_LIST"
  --val_dataset_path "$DATA"
  --val_data_list_path "$VAL_LIST"
  --model_param_path "$SAVE_ROOT"
  --crop_size 640
  --train_batch_size 2
  --eval_batch_size 2
  --num_workers 2
  --max_iters 120000
  --learning_rate 1e-4
  --weight_decay 5e-3
  --use_lovasz
  --ce_weight 1.0
  --lovasz_weight 1.0
  --roi_mode dilated_soft
  --roi_fusion multiplicative
  --roi_beta 1.0
  --roi_alpha 0.1
  --roi_dilate 5
  --use_mddm
  --mddm_variant base
  --mddm_gamma_init 0.0
  --mddm_use_roi
  --use_damage_exist_head
  --lambda_exist 0.1
  --save_policy smart
)

run_exp() {
  local exp_name="$1"
  local seed="$2"
  shift 2

  local exp_root="$SAVE_ROOT/$exp_name"
  local log_file="$LOG_ROOT/${exp_name}.log"
  mkdir -p "$exp_root"

  echo "============================================================"
  echo "Experiment : $exp_name"
  echo "Seed       : $seed"
  echo "GPU_ID     : $GPU_ID"
  echo "Save root  : $exp_root"
  echo "Log file   : $log_file"
  echo "Extra args : $*"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$TRAIN_PY" \
    --dataset "$exp_name" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$log_file"
}

run_exp "A11_ExistOnly_seed42" 42

run_exp "A11_ExistOnly_seed2026" 2026

run_exp "A11_Cons005_seed3407" 3407 \
  --use_exist_consistency \
  --lambda_exist_cons 0.05 \
  --exist_cons_warmup 10000 \
  --exist_cons_mode mse_detach

run_exp "A11_Cons01_seed3407" 3407 \
  --use_exist_consistency \
  --lambda_exist_cons 0.10 \
  --exist_cons_warmup 10000 \
  --exist_cons_mode mse_detach

echo "All A11_next experiments finished."
