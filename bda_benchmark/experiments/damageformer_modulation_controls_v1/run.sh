#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
M=bda_benchmark.experiments.damageformer_modulation_controls_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  python -m unittest "$M.test_controls" -v
  python -m "$M.train" --help
  exit 0
fi
test -s "${FRONT_CKPT:?Set fixed V2 front checkpoint}"
: "${OUTPUT_ROOT:?Set NEW output directory}"
COMMON=(--checkpoint "$FRONT_CKPT" --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
 --output "$OUTPUT_ROOT" --batch "${BATCH:-2}" --accum "${ACCUM:-8}"
 --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}")
case "$MODE" in
 smoke) python -u -m "$M.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke ;;
 train) python -u -m "$M.train" "${COMMON[@]}" --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}" ;;
 *) echo 'Use preflight|smoke|train'; exit 2 ;;
esac
