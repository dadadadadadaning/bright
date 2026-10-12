#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py"
DATA="$ROOT/bda_benchmark/dataset"
TRAIN_LIST="$DATA/splitname/standard_ML/train_set.txt"
VAL_LIST="$DATA/splitname/standard_ML/val_set.txt"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/saved_weights/unified_damage_guidance}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/unified_damage_guidance}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$LOG_ROOT"

COMMON_ARGS=(
  --train_dataset_path "$DATA"
  --train_data_list_path "$TRAIN_LIST"
  --val_dataset_path "$DATA"
  --val_data_list_path "$VAL_LIST"
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
  --val_interval 1000
  --val_max_batches 0
  --save_policy smart
  --model_param_path "$SAVE_ROOT"
)

A4_ARGS=(
  --roi_mode dilated_soft
  --roi_fusion multiplicative
  --roi_beta 1.0
  --roi_alpha 0.1
  --roi_dilate 5
  --use_mddm
  --mddm_variant base
  --mddm_gamma_init 0.0
  --mddm_use_roi
)

run_exp() {
  local name="$1"
  local seed="$2"
  shift 2
  local log_file="$LOG_ROOT/${name}_seed${seed}.log"
  echo "Launching $name seed=$seed on CUDA_VISIBLE_DEVICES=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" nohup python "$PY" \
    --dataset "${name}_seed${seed}" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "$log_file" 2>&1 &
}

SEEDS=(3407 42 2026)

for seed in "${SEEDS[@]}"; do
  run_exp "M03_A4_MDDMbase_ROId5" "$seed" "${A4_ARGS[@]}"
  run_exp "A11_ExistOnly" "$seed" "${A4_ARGS[@]}" --use_damage_exist_head --lambda_exist 0.1
  run_exp "A11_Cons005" "$seed" "${A4_ARGS[@]}" --use_damage_exist_head --lambda_exist 0.1 --use_exist_consistency --lambda_exist_cons 0.05 --exist_cons_warmup 10000 --exist_cons_mode mse_detach
  run_exp "A11_Calib01" "$seed" "${A4_ARGS[@]}" --use_damage_exist_head --lambda_exist 0.1 --use_exist_logit_calib --exist_calib_alpha 0.1 --exist_calib_beta 0.1 --exist_calib_warmup 10000
  run_exp "A11_FM01" "$seed" "${A4_ARGS[@]}" --use_damage_exist_head --lambda_exist 0.1 --use_exist_feature_modulation --exist_fm_gamma 0.1 --exist_fm_warmup 10000
done

echo "All jobs launched. Logs: $LOG_ROOT"
