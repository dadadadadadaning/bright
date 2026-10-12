#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU_ID="${1:-${GPU_ID:-0}}"
SPLIT="${2:-test}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/bda_benchmark:${PYTHONPATH:-}"

POST_PY="$ROOT/scripts/postprocess/postprocess_instance_superpixel_consistency.py"
SUM_PY="$ROOT/scripts/eval/summarize_E1_postprocess_consistency.py"
DATA="$ROOT/bda_benchmark/dataset"
DATA_LIST="$DATA/splitname/standard_ML/${SPLIT}_set.txt"
E1_DIR="$ROOT/saved_weights/BGCA_final/E1_Sampler_A11SHU_C640_s3407"
OUT_ROOT="$ROOT/eval_results_official/E1_postprocess_consistency"

mkdir -p "$OUT_ROOT"

CKPT="${E1_CKPT:-best_balanced_score.pth}"
MODEL_PATH="$E1_DIR/$CKPT"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: missing E1 checkpoint: $MODEL_PATH"
  exit 1
fi

BASE_ARGS=(
  --eval_protocol official
  --model_type DamageFormer_BGCA
  --model_path "$MODEL_PATH"
  --dataset_path "$DATA"
  --data_list_path "$DATA_LIST"
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

run_cfg() {
  local mode="$1"
  local dominant="$2"
  local min_area="$3"
  local keep="$4"
  local nseg="$5"
  local comp="$6"
  local name="${mode}_dom${dominant}_area${min_area}_keep${keep}_seg${nseg}_comp${comp}"
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  echo "CMD: CUDA_VISIBLE_DEVICES=$GPU_ID python $POST_PY ${BASE_ARGS[*]} --out_dir $out_dir --mode $mode --dominant_ratio $dominant --min_area $min_area --damage_keep_ratio $keep --n_segments $nseg --compactness $comp"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$POST_PY" \
    "${BASE_ARGS[@]}" \
    --out_dir "$out_dir" \
    --mode "$mode" \
    --dominant_ratio "$dominant" \
    --min_area "$min_area" \
    --damage_keep_ratio "$keep" \
    --n_segments "$nseg" \
    --compactness "$comp" \
    2>&1 | tee "$out_dir/eval.log"
}

for dominant in 0.55 0.65 0.75; do
  for min_area in 20 50 100; do
    run_cfg cc "$dominant" "$min_area" 0.25 1000 10
  done
done

for nseg in 500 1000 1500; do
  for comp in 10 20; do
    run_cfg slic 0.65 50 0.25 "$nseg" "$comp"
  done
done

python "$SUM_PY" --root "$OUT_ROOT"

echo "E1 postprocess consistency grid search finished."
