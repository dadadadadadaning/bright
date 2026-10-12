#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [[ "${1:-preflight}" == preflight ]]; then
 python -m unittest bda_benchmark.experiments.damageformer_srac_approx_v1.test_modules -v
 exit 0
fi
if [[ "${1:-}" != train ]]; then echo 'Usage: run.sh preflight|train [s0_grid|s1_learned|s2_modulated|s3_weighted]'; exit 2; fi
: "${A1_CKPT:?Set A1_CKPT}"
: "${OUTPUT_ROOT:?Set NEW OUTPUT_ROOT}"
python -u -m bda_benchmark.experiments.damageformer_srac_approx_v1.train \
 --checkpoint "$A1_CKPT" --dataset "$ROOT/bda_benchmark/dataset" \
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt" \
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt" \
 --output "$OUTPUT_ROOT" --variant "${2:-s3_weighted}" --steps "${STEPS:-5000}" \
 --eval_every "${EVAL_EVERY:-500}" --crop "${CROP:-512}" --workers "${NUM_WORKERS:-4}" \
 --stride "${REGION_STRIDE:-8}" --seed "${SEED:-3407}" --device "${DEVICE:-cuda:0}"
