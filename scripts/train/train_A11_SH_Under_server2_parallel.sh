#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU0="${1:-0}"
GPU1="${2:-1}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

TRAIN_PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py"
DATA="$ROOT/bda_benchmark/dataset"
TRAIN_LIST="$DATA/splitname/standard_ML/train_set.txt"
VAL_LIST="$DATA/splitname/standard_ML/val_set.txt"
SAVE_ROOT="$ROOT/saved_weights/BGCA_final"
LOG_ROOT="$ROOT/logs/train_A11_SH_Under"

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --train_dataset_path "$DATA"
  --train_data_list_path "$TRAIN_LIST"
  --val_dataset_path "$DATA"
  --val_data_list_path "$VAL_LIST"
  --model_param_path "$SAVE_ROOT"
  --crop_size 640
  --train_batch_size 2
  --eval_batch_size 1
  --num_workers 4
  --max_iters 120000
  --seed 3407
  --learning_rate 1e-4
  --weight_decay 5e-3
  --use_lovasz
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
  --use_exist_consistency
  --lambda_exist_cons 0.05
  --exist_cons_warmup 20000
  --exist_cons_mode mse_detach
  --save_policy smart
)

run_exp() {
  local gpu_id="$1"
  local exp_name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu_id" python "$TRAIN_PY" \
    --dataset "$exp_name" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$LOG_ROOT/${exp_name}.log"
}

run_exp "$GPU0" "A11_Cons005_w20000_SH_s005_seed3407_C640_I120k_BS2" \
  --use_severity_head \
  --lambda_severity 0.05 \
  --severity_warmup 20000 &
PID0=$!

run_exp "$GPU1" "A11_Cons005_w20000_SH_s005_Under005_seed3407_C640_I120k_BS2" \
  --use_severity_head \
  --lambda_severity 0.05 \
  --severity_warmup 20000 \
  --use_damage_under_loss \
  --lambda_under 0.005 \
  --under_warmup 20000 \
  --under_margin 0.05 &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "One or more experiments failed: status0=$STATUS0 status1=$STATUS1"
  exit 1
fi
