#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/BRIGHT-master"
GPU_IDS="${1:-${GPU_IDS:-0,1}}"
EXP_NAME="BD_PGFNet_OfficialStrict_DP_seed3407_C640_BS16_50kSteps"
SAVE_ROOT="$ROOT/saved_weights/official_strict_dp/BD_PGFNet"
LOG_ROOT="$ROOT/logs/train_official_strict/BD_PGFNet"
TRAIN_PY="$ROOT/bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06_dp.py"
DATA_ROOT="$ROOT/bda_benchmark/dataset"

cd "$ROOT"
mkdir -p "$SAVE_ROOT" "$SAVE_ROOT/$EXP_NAME" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

HELP_TXT="$(python "$TRAIN_PY" --help 2>&1 || true)"
has_arg() { grep -q -- "$1" <<< "$HELP_TXT"; }
require_arg() {
  if ! has_arg "$1"; then
    echo "[ERROR] required argument is not supported by $TRAIN_PY: $1" >&2
    exit 2
  fi
}
add_arg() {
  local opt="$1"
  local val="$2"
  if has_arg "$opt"; then
    CMD+=("$opt" "$val")
  else
    echo "[SKIP] unsupported argument: $opt"
  fi
}
add_flag() {
  local opt="$1"
  if has_arg "$opt"; then
    CMD+=("$opt")
  else
    echo "[SKIP] unsupported flag: $opt"
  fi
}

for opt in --crop_size --train_batch_size --max_iters --learning_rate --weight_decay --use_lovasz --use_damage_exist_head --use_bdpgffm; do
  require_arg "$opt"
done

CMD=(python "$TRAIN_PY")
add_arg --dataset BRIGHT
add_arg --seed 3407
add_arg --train_dataset_path "$DATA_ROOT"
add_arg --train_data_list_path "$DATA_ROOT/splitname/standard_ML/train_set.txt"
add_arg --val_dataset_path "$DATA_ROOT"
add_arg --val_data_list_path "$DATA_ROOT/splitname/standard_ML/val_set.txt"
add_arg --model_param_path "$SAVE_ROOT"
add_arg --exp_name "$EXP_NAME"
add_arg --device auto
add_arg --crop_size 640
add_arg --train_batch_size 16
add_arg --eval_batch_size 2
add_arg --num_workers 4
add_arg --max_iters 50000
add_arg --learning_rate 1e-4
add_arg --weight_decay 5e-3
add_arg --log_interval 20
add_arg --val_interval 1000
add_arg --val_max_batches 0
add_arg --save_policy smart
add_flag --use_lovasz

add_arg --roi_mode dilated_soft
add_arg --roi_fusion multiplicative
add_arg --roi_beta 1.0
add_arg --roi_alpha 0.1
add_arg --roi_dilate 5
add_flag --use_mddm
add_arg --mddm_variant base
add_arg --mddm_gamma_init 0.0
add_flag --mddm_use_roi

add_flag --use_damage_exist_head
add_arg --lambda_exist 0.05
add_arg --exist_pos_weight 2.0
add_arg --exist_loss_type bce

add_flag --use_bdpgffm
add_arg --bdpgffm_gamma 0.2
add_flag --bdpgffm_detach_prior
add_arg --bdpgffm_dropout 0.0

add_flag --use_region_pool_aux
add_arg --lambda_region 0.03
add_arg --region_purity_thr 0.70
add_arg --region_min_pixels 8
add_arg --region_feat_dim 256
add_arg --region_hidden_dim 128
add_arg --region_warmup 10000
if has_arg --region_class_weights; then
  CMD+=(--region_class_weights 1.0 1.5 2.0)
fi

echo "[INFO] Formal experiment: $EXP_NAME"
echo "[INFO] GPU_IDS=$GPU_IDS"
echo "[INFO] max_iters=50000 optimizer steps"
echo "[INFO] Log: $LOG_ROOT/${EXP_NAME}.log"
echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee "$LOG_ROOT/${EXP_NAME}.log"
