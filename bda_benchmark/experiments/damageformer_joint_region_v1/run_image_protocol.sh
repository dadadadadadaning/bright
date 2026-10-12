#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_joint_region_v1
mode="${1:-preflight}"
if [[ "$mode" == preflight ]]; then
 python -m unittest "$MODULE.test_model" -v
 python -m "$MODULE.train_image_protocol" --help
 exit 0
fi
test -s "${SWIN_PRETRAIN:?Set real torchvision Swin-T weight path}"
run_one() {
 local variant="$1"
 local steps=50000
 local extra=()
 if [[ "$mode" == smoke ]]; then steps=20; extra=(--smoke); fi
 python -u -m "$MODULE.train_image_protocol" \
 --variant "$variant" --pretrain "$SWIN_PRETRAIN" \
 --dataset "$ROOT/bda_benchmark/dataset" \
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt" \
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt" \
 --output "${OUTPUT_ROOT:?Set NEW output root}/$variant" \
 --steps "$steps" --eval_every "${EVAL_EVERY:-1000}" \
 --batch "${BATCH:-4}" --accum "${ACCUM:-4}" --workers "${WORKERS:-4}" \
 --seed "${SEED:-3407}" --amp "${AMP:-bf16}" --device "${DEVICE:-cuda:0}" \
 --schedule "${SCHEDULE:-warmup_cosine}" "${extra[@]}"
}
case "$mode" in
 smoke|train) run_one "${2:-t0}" ;;
 sequence) for variant in t0 t1 t3; do run_one "$variant"; done ;;
 *) echo 'Use preflight | smoke VARIANT | train VARIANT | sequence' >&2; exit 2 ;;
esac
