#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_iterative_partition_v1
mode="${1:-preflight}"
if [[ "$mode" == preflight ]]; then
  python -m unittest "$MODULE.test_model" -v
  python -m "$MODULE.train" --help
  echo 'Partition V1 preflight passed. No Test access. GPU training not checked.'
  exit 0
fi
case "$mode" in smoke|train|sequence|smoke_sequence) ;; *) echo 'Use preflight | smoke VARIANT | train VARIANT | sequence | smoke_sequence'; exit 2;; esac
: "${SWIN_PRETRAIN:?Set SWIN_PRETRAIN to real torchvision Swin-T weights, NOT A1 task weights}"
test -s "$SWIN_PRETRAIN"
: "${OUTPUT_ROOT:?Set a NEW output root}"
run_one() {
  local variant="$1"
  local steps=50000
  local extra=()
  if [[ "$mode" == smoke || "$mode" == smoke_sequence ]]; then steps=20; extra=(--smoke --smoke_val_images 2); fi
  python -u -m "$MODULE.train" \
    --variant "$variant" --pretrain "$SWIN_PRETRAIN" \
    --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}" \
    --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}" \
    --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}" \
    --output "$OUTPUT_ROOT/$variant" --steps "$steps" --eval_every "${EVAL_EVERY:-1000}" \
    --batch "${BATCH:-4}" --accum "${ACCUM:-4}" --workers "${WORKERS:-4}" \
    --dim 64 --stride "${STRIDE:-8}" --temperature "${TEMPERATURE:-0.2}" --spatial_weight "${SPATIAL_WEIGHT:-0.1}" \
    --lr 0.0001 --warmup 1000 --schedule warmup_cosine \
    --seed "${SEED:-3407}" --device "${DEVICE:-cuda:0}" --amp "${AMP:-bf16}" "${extra[@]}"
}
case "$mode" in
  smoke|train) run_one "${2:-iter3}" ;;
  sequence|smoke_sequence)
    read -r -a variants <<< "${VARIANTS:-single iter1 iter3 grid}"
    for variant in "${variants[@]}"; do
      case "$variant" in pixel|single|iter1|iter3|grid) ;; *) echo "Unknown variant: $variant"; exit 2;; esac
      if [[ -e "$OUTPUT_ROOT/$variant" ]]; then echo "Refusing existing output: $OUTPUT_ROOT/$variant"; exit 2; fi
    done
    for variant in "${variants[@]}"; do
      echo "===== $variant START $(date -Iseconds) ====="
      run_one "$variant"
      echo "===== $variant COMPLETE $(date -Iseconds) ====="
    done ;;
esac
