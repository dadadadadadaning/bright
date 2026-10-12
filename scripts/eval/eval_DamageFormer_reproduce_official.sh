#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${1:-${GPU_ID:-0}}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

EVAL_PY="$ROOT/full_evaluate_unified.py"
DATA="$ROOT/bda_benchmark/dataset"
TEST_LIST="$DATA/splitname/standard_ML/test_set.txt"
SAVE_ROOT="$ROOT/saved_weights/BGCA_final"
OUT_ROOT="$ROOT/eval_results_official/DamageFormer_Reproduce"
EXP_NAME="DamageFormer_Reproduce_seed3407_C640_I120k_BS2"

mkdir -p "$OUT_ROOT"

CHECKPOINTS=(
  best_damage_score.pth
  best_valid_miou.pth
  best_balanced_score.pth
  best_damaged_iou.pth
  best_destroyed_iou.pth
  latest_resume.pth
)

find_latest_exp_dir() {
  local exp_root="$SAVE_ROOT/$EXP_NAME"
  if [[ ! -d "$exp_root" ]]; then
    echo ""
    return 0
  fi
  find "$exp_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

EXP_DIR="$(find_latest_exp_dir)"
if [[ -z "$EXP_DIR" || ! -d "$EXP_DIR" ]]; then
  echo "WARNING: no trained experiment directory found for $EXP_NAME under $SAVE_ROOT; skip."
  exit 0
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  MODEL_PATH="$EXP_DIR/$ckpt"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "WARNING: missing checkpoint $MODEL_PATH; skip."
    continue
  fi

  CKPT_NAME="${ckpt%.pth}"
  SAVE_DIR="$OUT_ROOT/$EXP_NAME/$CKPT_NAME"
  mkdir -p "$SAVE_DIR"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_PY" \
    --eval_protocol official \
    --model_type DamageFormer_BGCA \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATA" \
    --data_list_path "$TEST_LIST" \
    --save_dir "$SAVE_DIR" \
    --image_size 640 \
    --batch_size 1 \
    --num_workers 0 \
    --roi_mode none \
    2>&1 | tee "$SAVE_DIR/eval.log"
done

python "$ROOT/scripts/eval/analyze_DamageFormer_reproduce_full.py"

echo "DamageFormer reproduce official evaluation finished."
