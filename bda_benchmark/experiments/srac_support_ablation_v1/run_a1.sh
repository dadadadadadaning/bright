#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-preflight}"
export SOURCES=fused_roi
export TRAIN_RECORDS="${A1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SRAC_SUPPORT_ABLATION_V1/damageformer_a1/cache}/train.pt"
export VAL_RECORDS="${A1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SRAC_SUPPORT_ABLATION_V1/damageformer_a1/cache}/validation.pt"
build() {
  local split="$1" count="$2" list="$3" output="$4"
  : "${A1_CKPT:?Set A1_CKPT to verified compatible A1 checkpoint}"
  local extra=()
  if [[ "$split" == validation && "$count" == 450 ]]; then
    : "${EXPECTED_A1_MIOU:?Set verified foreground Val mIoU as fraction, not percent}"
    extra+=(--expected_miou "$EXPECTED_A1_MIOU")
  fi
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.srac_support_ablation_v1.build_a1_records \
    --checkpoint "$A1_CKPT" --dataset "${DATASET_PATH:-$ROOT/bda_benchmark/dataset}" \
    --data_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/$list" --split "$split" \
    --output "$output" --expected_images "$count" --max_images "${MAX_IMAGES:-0}" \
    --workers "${NUM_WORKERS:-4}" --device "${DEVICE:-cuda:0}" "${extra[@]}"
}
case "$MODE" in
  preflight) bash "$HERE/run.sh" preflight ;;
  smoke_cache) MAX_IMAGES=2 build validation 2 val_set.txt "$(dirname "$VAL_RECORDS")/smoke.pt" ;;
  build_val) build validation 450 val_set.txt "$VAL_RECORDS" ;;
  build_train) build train 2918 train_set.txt "$TRAIN_RECORDS" ;;
  all) bash "$HERE/run.sh" all ;;
  *) echo 'Usage: run_a1.sh preflight|smoke_cache|build_val|build_train|all'; exit 2 ;;
esac
