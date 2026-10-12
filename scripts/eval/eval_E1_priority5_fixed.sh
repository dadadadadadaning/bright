#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${1:-${GPU_ID:-0}}"
SPLIT="${2:-test}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

EVAL_PY="$ROOT/full_evaluate_unified.py"
DATA="$ROOT/bda_benchmark/dataset"
DATA_LIST="$DATA/splitname/standard_ML/${SPLIT}_set.txt"
SAVE_ROOT="$ROOT/saved_weights/BGCA_final"
OUT_ROOT="$ROOT/eval_results_official/E1_priority5_fixed_test"

mkdir -p "$OUT_ROOT"

CHECKPOINTS=(
  best_balanced_score.pth
  best_valid_miou.pth
  best_damage_score.pth
  best_destroyed_iou.pth
  best_damaged_iou.pth
  latest_resume.pth
)

BASE_ARGS=(
  --eval_protocol official
  --model_type DamageFormer_BGCA
  --dataset_path "$DATA"
  --data_list_path "$DATA_LIST"
  --image_size 640
  --batch_size 1
  --num_workers 0
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
  --use_severity_head
  --lambda_severity 0.1
  --use_damage_under_loss
  --use_lovasz
)

CONS_ARGS=(
  --use_exist_consistency
  --lambda_exist_cons 0.05
  --exist_cons_warmup 20000
  --exist_cons_mode mse_detach
)

resolve_exp_dir() {
  local exp_name="$1"
  local direct="$SAVE_ROOT/$exp_name"
  if [[ -d "$direct" ]]; then
    echo "$direct"
    return 0
  fi
  local nested
  nested="$(find "$SAVE_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$exp_name" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$nested" ]]; then
    echo "$nested"
    return 0
  fi
  echo ""
}

check_load_status() {
  local log_path="$1"
  local missing unexpected
  missing="$(grep -E '^Missing keys:' "$log_path" | tail -n 1 | awk '{print $3}' || true)"
  unexpected="$(grep -E '^Unexpected keys:' "$log_path" | tail -n 1 | awk '{print $3}' || true)"
  missing="${missing:-NA}"
  unexpected="${unexpected:-NA}"
  if [[ "$missing" != "0" || "$unexpected" != "0" ]]; then
    echo "WARNING: load key mismatch in $log_path: Missing keys=$missing, Unexpected keys=$unexpected"
    echo "WARNING: this result should not be used until model args/checkpoint compatibility is fixed." >> "$log_path"
  fi
}

eval_exp() {
  local exp_name="$1"
  local use_cons="$2"
  local exp_dir
  exp_dir="$(resolve_exp_dir "$exp_name")"
  if [[ -z "$exp_dir" || ! -d "$exp_dir" ]]; then
    echo "WARNING: no trained experiment directory found for $exp_name under $SAVE_ROOT; skip."
    return 0
  fi

  echo "============================================================"
  echo "Evaluating : $exp_name"
  echo "Cons args  : $use_cons"
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
    echo "CMD: CUDA_VISIBLE_DEVICES=$GPU_ID python $EVAL_PY ${BASE_ARGS[*]} --model_path $model_path --save_dir $save_dir"
    if [[ "$use_cons" == "yes" ]]; then
      echo "CMD extra: ${CONS_ARGS[*]}"
      CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_PY" \
        "${BASE_ARGS[@]}" "${CONS_ARGS[@]}" \
        --model_path "$model_path" \
        --save_dir "$save_dir" \
        2>&1 | tee "$save_dir/eval.log"
    else
      CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_PY" \
        "${BASE_ARGS[@]}" \
        --model_path "$model_path" \
        --save_dir "$save_dir" \
        2>&1 | tee "$save_dir/eval.log"
    fi
    check_load_status "$save_dir/eval.log"
  done
}

eval_exp "E1_Sampler_A11SHU_C640_s3407" "yes"
eval_exp "E1_noCons_Sampler_A11SHU_C640_s3407" "no"
eval_exp "E1_strongerSampler_A11SHU_C640_s3407" "yes"
eval_exp "E1_noCons_strongerSampler_A11SHU_C640_s3407" "no"

python "$ROOT/scripts/eval/analyze_E1_priority5.py"

echo "E1 priority5 fixed evaluation finished."
