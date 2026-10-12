#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
M=bda_benchmark.experiments.damageformer_roi_partition_control_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  python -m unittest "$M.test_core" -v
  python -m "$M.train" --help
  exit 0
fi
case "$MODE" in smoke|train) ;; *) echo 'Use preflight|smoke|train'; exit 2 ;; esac
test -s "${FRONT_CKPT:?Set fixed V2 front snapshot, NOT A1 or readout weights}"
: "${OUTPUT_ROOT:?Set new non-existing output directory}"
read -r -a MODES <<< "${VARIANTS:-s0 s1 s2 c2}"
COMMON=(--checkpoint "$FRONT_CKPT" --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
 --output "$OUTPUT_ROOT" --variants "${MODES[@]}"
 --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --workers "${WORKERS:-4}"
 --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}"
 --soft_strength "${SOFT_STRENGTH:-1}" --threshold "${ROI_THRESHOLD:-0.5}")
if [[ "$MODE" == smoke ]]; then
  python -u -m "$M.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke
else
  python -u -m "$M.train" "${COMMON[@]}" --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}"
fi
