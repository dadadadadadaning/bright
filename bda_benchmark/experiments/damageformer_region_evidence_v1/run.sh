#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
M=bda_benchmark.experiments.damageformer_region_evidence_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
 python -m unittest "$M.test_evidence" -v
 python -m "$M.audit" --help
 python -m "$M.train_heads" --help
 exit 0
fi
test -s "${FRONT_CKPT:?Set fixed V2 checkpoint}"
: "${OUTPUT_ROOT:?Set new isolated output directory}"
COMMON=(--checkpoint "$FRONT_CKPT" --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
 --output "$OUTPUT_ROOT" --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}")
case "$MODE" in
 audit) python -u -m "$M.audit" "${COMMON[@]}" ;;
 smoke_audit) python -u -m "$M.audit" "${COMMON[@]}" --smoke ;;
 train_heads) python -u -m "$M.train_heads" "${COMMON[@]}" --steps "${PROBE_STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}" --batch "${BATCH:-2}" --accum "${ACCUM:-8}" ;;
 smoke_heads) python -u -m "$M.train_heads" "${COMMON[@]}" --steps 20 --eval_every 20 --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --smoke ;;
 *) echo 'Use preflight|smoke_audit|audit|smoke_heads|train_heads'; exit 2;;
esac
