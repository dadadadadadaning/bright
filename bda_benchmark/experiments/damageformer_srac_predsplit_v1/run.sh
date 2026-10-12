#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_srac_predsplit_v1
mode="${1:-preflight}"
if [[ "$mode" == preflight ]]; then
    python -m unittest "$MODULE.test_modules" -v
    exit 0
fi
run_one() {
    local variant="$1"
    python -u -m "$MODULE.train" \
      --checkpoint "${A1_CKPT:?Set A1_CKPT}" \
      --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}" \
      --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}" \
      --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}" \
      --output "${OUTPUT_ROOT:?Set a NEW OUTPUT_ROOT}/$variant" \
      --variant "$variant" --steps "${STEPS:-5000}" \
      --eval_every "${EVAL_EVERY:-500}" --stride 8 --crop 512 \
      --workers "${WORKERS:-4}" --seed "${SEED:-3407}" --device "${DEVICE:-cuda:0}"
}
case "$mode" in
 train) run_one "${2:?Supply variant}" ;;
 sequence)
    python -m unittest "$MODULE.test_modules" -v
    for variant in s3_predsplit grid_predsplit; do
        run_one "$variant"
    done
    echo 'PRED SPLIT SEQUENCE COMPLETE'
    ;;
 *) echo 'Use preflight | train VARIANT | sequence' >&2; exit 2 ;;
esac
