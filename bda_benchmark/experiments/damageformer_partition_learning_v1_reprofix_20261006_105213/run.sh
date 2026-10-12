#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_partition_learning_v1
PACKAGE_DIR="$ROOT/bda_benchmark/experiments/damageformer_partition_learning_v1"
MODE="${1:-preflight}"

if (( $# > 1 )); then
  echo 'Supply one mode only; configure paths through environment variables.' >&2
  exit 2
fi
case "$MODE" in
  preflight|report|smoke_structure|structure|smoke_multimodal|multimodal|smoke_support|support) ;;
  *) echo 'Use preflight|smoke_structure|structure|smoke_multimodal|multimodal|smoke_support|support|report' >&2; exit 2 ;;
esac
: "${PYTHON:?Set an absolute Python executable path}"
if [[ "$PYTHON" != /* || ! -x "$PYTHON" ]]; then
  echo "PYTHON must be an absolute executable path: $PYTHON" >&2
  exit 2
fi

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest discover -s "$PACKAGE_DIR" -t "$ROOT" -p 'test_*.py' -v
  "$PYTHON" -m "$MODULE.train" --help
  exit 0
fi
: "${OUTPUT_ROOT:?Set the output directory}"
if [[ "$MODE" == report ]]; then
  "$PYTHON" -m "$MODULE.report" --root "$OUTPUT_ROOT"
  exit 0
fi

: "${FRONT_CKPT:?Set the immutable front snapshot matching the source run}"
: "${SOURCE_RUN:?Set a completed formal ROI_EXCHANGE_V1 training directory}"
for INPUT in "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/manifest.json" "$SOURCE_RUN/summary.json"; do
  if [[ ! -s "$INPUT" ]]; then
    echo "Required input missing or empty: $INPUT" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]]; then
  echo "Output exists; choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

SUITE="${MODE#smoke_}"
COMMON=(--suite "$SUITE" --checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN"
  --source_head hard --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT" --batch "${BATCH:-2}" --accum "${ACCUM:-8}"
  --workers "${WORKERS:-4}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}"
  --device "${DEVICE:-cuda:0}" --structure_weight "${STRUCTURE_WEIGHT:-0.1}"
  --sar_weight "${SAR_WEIGHT:-0.2}" --support_temperature "${SUPPORT_TEMPERATURE:-0.2}")
# Stale VARIANTS, EXP, STEPS and EVAL_EVERY never select groups or alter this protocol.
if [[ "$MODE" == smoke_* ]]; then
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke
else
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --steps 5000 --eval_every 500
fi
"$PYTHON" -m "$MODULE.report" --root "$OUTPUT_ROOT"
