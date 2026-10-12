#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
DATA="$ROOT/bda_benchmark/dataset"
TEST_LIST="$DATA/splitname/standard_ML/test_set.txt"
WEIGHT_ROOT="${WEIGHT_ROOT:-$ROOT/saved_weights/unified_damage_guidance}"
OUT_ROOT="${OUT_ROOT:-$ROOT/eval_results/unified_damage_guidance_official}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_ROOT"

CHECKPOINTS=(
  best_damage_score.pth
  best_valid_miou.pth
  best_damaged_iou.pth
  best_destroyed_iou.pth
  latest_resume.pth
)

BASE_EVAL_ARGS=(
  --eval_protocol official
  --model_type DamageFormer_BGCA
  --dataset_path "$DATA"
  --data_list_path "$TEST_LIST"
  --image_size 640
  --batch_size 1
  --num_workers 2
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

find_latest_exp_dir() {
  local dataset="$1"
  find "$WEIGHT_ROOT/$dataset" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

eval_exp() {
  local model_name="$1"
  local seed="$2"
  shift 2
  local dataset="${model_name}_seed${seed}"
  local exp_dir
  exp_dir="$(find_latest_exp_dir "$dataset")"
  if [[ -z "$exp_dir" ]]; then
    echo "Missing experiment directory for $dataset; skip."
    return 0
  fi
  echo "Evaluating $dataset from $exp_dir"
  for ckpt in "${CHECKPOINTS[@]}"; do
    local model_path="$exp_dir/$ckpt"
    if [[ ! -f "$model_path" ]]; then
      echo "  Missing checkpoint $ckpt; skip."
      continue
    fi
    local save_dir="$OUT_ROOT/$model_name/seed${seed}/${ckpt%.pth}"
    mkdir -p "$save_dir"
    CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/full_evaluate_unified.py" \
      "${BASE_EVAL_ARGS[@]}" \
      --model_path "$model_path" \
      --save_dir "$save_dir" \
      "$@" \
      2>&1 | tee "$save_dir/eval.log"
  done
}

SEEDS=(3407 42 2026)

for seed in "${SEEDS[@]}"; do
  eval_exp "M03_A4_MDDMbase_ROId5" "$seed"
  eval_exp "A11_ExistOnly" "$seed" --use_damage_exist_head --lambda_exist 0.1
  eval_exp "A11_Cons005" "$seed" --use_damage_exist_head --lambda_exist 0.1
  eval_exp "A11_Calib01" "$seed" --use_damage_exist_head --lambda_exist 0.1 --use_exist_logit_calib --exist_calib_alpha 0.1 --exist_calib_beta 0.1 --exist_calib_warmup 10000
  eval_exp "A11_FM01" "$seed" --use_damage_exist_head --lambda_exist 0.1 --use_exist_feature_modulation --exist_fm_gamma 0.1 --exist_fm_warmup 10000
done

python "$ROOT/scripts/eval/summarize_unified_official_results.py" --eval_root "$OUT_ROOT" --out_dir "$OUT_ROOT"
