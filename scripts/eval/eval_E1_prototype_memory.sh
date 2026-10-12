#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${1:-${GPU_ID:-0}}"
SPLIT="${2:-test}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

BUILD_PY="$ROOT/scripts/prototype/build_damage_prototype_bank.py"
EVAL_PY="$ROOT/scripts/prototype/eval_E1_prototype_refine.py"
SUM_PY="$ROOT/scripts/eval/summarize_E1_prototype_memory.py"
DATA="$ROOT/bda_benchmark/dataset"
TRAIN_LIST="$DATA/splitname/standard_ML/train_set.txt"
TEST_LIST="$DATA/splitname/standard_ML/${SPLIT}_set.txt"
E1_DIR="$ROOT/saved_weights/BGCA_final/E1_Sampler_A11SHU_C640_s3407"
OUT_ROOT="$ROOT/eval_results_official/E1_prototype_memory"

mkdir -p "$OUT_ROOT"

CKPT="${E1_CKPT:-best_balanced_score.pth}"
MODEL_PATH="$E1_DIR/$CKPT"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: missing E1 checkpoint: $MODEL_PATH"
  exit 1
fi

BASE_MODEL_ARGS=(
  --eval_protocol official
  --model_type DamageFormer_BGCA
  --image_size 640
  --batch_size 1
  --num_workers 0
  --save_dir "$OUT_ROOT/base_parser_compat"
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
  --use_severity_head
  --lambda_severity 0.1
  --use_damage_under_loss
  --use_lovasz
)

for k in 1 5; do
  BANK_DIR="$OUT_ROOT/proto_k${k}"
  mkdir -p "$BANK_DIR"
  echo "Building prototype bank K=$k"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$BUILD_PY" \
    "${BASE_MODEL_ARGS[@]}" \
    --checkpoint "$MODEL_PATH" \
    --dataset_root "$DATA" \
    --train_list "$TRAIN_LIST" \
    --save_path "$BANK_DIR" \
    --num_clusters "$k" \
    --max_pixels_per_class 200000 \
    --classes damaged destroyed \
    --feature_layer penultimate \
    2>&1 | tee "$BANK_DIR/build.log"

  for alpha in 0.05 0.1 0.2 0.3; do
    for region in building uncertain_damage; do
      CFG_DIR="$OUT_ROOT/k${k}_a${alpha}_${region}"
      mkdir -p "$CFG_DIR"
      echo "Evaluating prototype refine K=$k alpha=$alpha region=$region"
      CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_PY" \
        "${BASE_MODEL_ARGS[@]}" \
        --checkpoint "$MODEL_PATH" \
        --prototype_path "$BANK_DIR/prototypes.pt" \
        --dataset_root "$DATA" \
        --test_list "$TEST_LIST" \
        --out_dir "$CFG_DIR" \
        --alpha "$alpha" \
        --topk 1 \
        --apply_region "$region" \
        --uncertainty_threshold 0.65 \
        2>&1 | tee "$CFG_DIR/eval.log"
    done
  done
done

python "$SUM_PY" --root "$OUT_ROOT"

echo "E1 prototype memory grid search finished."
