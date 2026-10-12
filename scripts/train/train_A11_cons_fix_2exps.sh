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
SAVE_ROOT="$ROOT/saved_weights/BGCA_clean/A11_cons_fix_2exps"
LOG_ROOT="$ROOT/logs/train_A11_next"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --train_dataset_path "$DATA"
  --train_data_list_path "$TRAIN_LIST"
  --val_dataset_path "$DATA"
  --val_data_list_path "$VAL_LIST"
  --model_param_path "$SAVE_ROOT"
  --image_size 640
  --train_batch_size 2
  --eval_batch_size 1
  --num_workers 0
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
  --exist_loss_type bce
  --exist_focal_gamma 2.0
  --save_policy smart
)

run_exp() {
  local exp_name="$1"
  local seed="$2"
  local lambda_exist_cons="$3"
  local warmup="$4"
  local mode="$5"

  local log_file="$LOG_ROOT/${exp_name}.log"

  echo "============================================================"
  echo "Experiment             : $exp_name"
  echo "Seed                   : $seed"
  echo "GPU_ID                 : $GPU_ID"
  echo "Save root              : $SAVE_ROOT/$exp_name"
  echo "Log file               : $log_file"
  echo "crop/image_size        : 640"
  echo "train/eval batch size  : 2 / 1"
  echo "num_workers            : 0"
  echo "max_iters              : 120000"
  echo "lambda_exist           : 0.1"
  echo "lambda_exist_cons      : $lambda_exist_cons"
  echo "exist_cons_warmup      : $warmup"
  echo "exist_cons_mode        : $mode"
  echo "ROI                    : dilated_soft, multiplicative, beta=1.0, alpha=0.1, dilate=5"
  echo "MDDM                   : base, gamma_init=0.0, use_roi=true"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$TRAIN_PY" \
    --dataset "$exp_name" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" \
    --use_exist_consistency \
    --lambda_exist_cons "$lambda_exist_cons" \
    --exist_cons_warmup "$warmup" \
    --exist_cons_mode "$mode" \
    2>&1 | tee "$log_file"
}

run_exp "A11_Cons002_w20000_seed3407_C640_I120k_BS2" 3407 0.02 20000 mse_detach
run_exp "A11_Cons005_w20000_seed3407_C640_I120k_BS2" 3407 0.05 20000 mse_detach

echo "All A11 consistency fix experiments finished."
