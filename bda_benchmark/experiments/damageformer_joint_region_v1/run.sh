#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_joint_region_v1
if [[ "${1:-preflight}" == preflight ]]; then
 python -m unittest "$MODULE.test_model" -v
 exit 0
fi
run_one() {
 python -u -m "$MODULE.train" --variant "$1" --pretrain "${SWIN_PRETRAIN:?Set torchvision Swin-T weight file}" \
 --dataset "$ROOT/bda_benchmark/dataset" \
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt" \
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt" \
 --output "${OUTPUT_ROOT:?Set NEW output root}/$1" \
 --steps "${STEPS:-40000}" --eval_every "${EVAL_EVERY:-1000}" \
 --batch "${BATCH:-1}" --accum "${ACCUM:-16}" --workers "${WORKERS:-4}" \
 --seed "${SEED:-3407}" --amp "${AMP:-bf16}" --device "${DEVICE:-cuda:0}"
}
case "$1" in
 train) run_one "${2:?t0/t1/t2/t3/t4}" ;;
 sequence) for variant in t0 t1 t3; do run_one "$variant"; done ;;
 *) echo 'Use preflight | train VARIANT | sequence' >&2; exit 2 ;;
esac
