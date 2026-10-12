#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${GPU_ID:-0}"

EVAL_PY="$ROOT/full_evaluate_unified.py"
DATA="$ROOT/bda_benchmark/dataset"
TEST_LIST="$DATA/splitname/standard_ML/test_set.txt"

SAVE_ROOT="$ROOT/saved_weights/BGCA_clean/A11_next"
OUT_ROOT="$ROOT/eval_results_official/A11_next"

mkdir -p "$OUT_ROOT"

CHECKPOINTS=(
  best_valid_miou.pth
  best_balanced_score.pth
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
  --use_damage_exist_head
  --lambda_exist 0.1
)

find_latest_exp_dir() {
  local exp_name="$1"
  local exp_root="$SAVE_ROOT/$exp_name"
  if [[ ! -d "$exp_root" ]]; then
    echo ""
    return 0
  fi
  find "$exp_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

eval_exp() {
  local exp_name="$1"
  local exp_dir
  exp_dir="$(find_latest_exp_dir "$exp_name")"

  if [[ -z "$exp_dir" ]]; then
    echo "WARNING: no trained experiment directory found for $exp_name under $SAVE_ROOT; skip."
    return 0
  fi

  echo "============================================================"
  echo "Evaluating : $exp_name"
  echo "Exp dir    : $exp_dir"
  echo "Output root: $OUT_ROOT/$exp_name"
  echo "============================================================"

  for ckpt in "${CHECKPOINTS[@]}"; do
    local model_path="$exp_dir/$ckpt"
    if [[ ! -f "$model_path" ]]; then
      echo "WARNING: missing checkpoint $model_path; skip."
      continue
    fi

    local ckpt_name="${ckpt%.pth}"
    local save_dir="$OUT_ROOT/$exp_name/$ckpt_name"
    mkdir -p "$save_dir"

    echo "Evaluating checkpoint: $model_path"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_PY" \
      "${BASE_EVAL_ARGS[@]}" \
      --model_path "$model_path" \
      --save_dir "$save_dir" \
      2>&1 | tee "$save_dir/eval.log"
  done
}

eval_exp "A11_ExistOnly_seed42"
eval_exp "A11_ExistOnly_seed2026"
eval_exp "A11_Cons005_seed3407"
eval_exp "A11_Cons01_seed3407"

python "$ROOT/scripts/eval/summarize_A11_next.py"

echo "A11_next official evaluation finished."
