#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_joint_context_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$MODULE.test_contracts" -v
  "$PYTHON" -m "$MODULE.train" --help >/dev/null
  echo "Joint-context contracts passed. Real BRIGHT/CUDA smoke is still required. Test is forbidden."
  exit 0
fi

case "$MODE" in smoke|train) ;; *) echo "Use preflight|smoke|train"; exit 2 ;; esac

: "${RUN_ROOT:?Set a NEW isolated RUN_ROOT}"
: "${A1_CKPT:?Set the exact A1 checkpoint referenced by FRONT_CKPT}"
: "${FRONT_CKPT:?Set a completed damageformer_srac_transformer_v1 front checkpoint}"
: "${CACHE:?Set its completed Train-only cache directory}"
: "${NEIGHBORS:?Set its immutable neighbor-table directory}"

DATASET="${DATASET:-$ROOT/bda_benchmark/dataset}"
TRAIN_LIST="$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
ARGS=()
STEPS=5000
EVAL_EVERY=500
if [[ "$MODE" == smoke ]]; then
  ARGS+=(--smoke)
  STEPS=20
  EVAL_EVERY=20
fi

"$PYTHON" -u -m "$MODULE.train" \
  --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
  --cache "$CACHE" --neighbors "$NEIGHBORS" \
  --dataset "$DATASET" --train_list "$TRAIN_LIST" --val_list "$VAL_LIST" \
  --output "$RUN_ROOT" --steps "$STEPS" --eval_every "$EVAL_EVERY" \
  --workers "${WORKERS:-4}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}" \
  --topk "${TOPK:-8}" --retrieval_policy "${RETRIEVAL_POLICY:-global}" \
  --exclusion "${EXCLUSION:-cross_event}" --device "${DEVICE:-cuda:0}" \
  --amp "${AMP:-bf16}" "${ARGS[@]}"

echo "COMPLETE $MODE. Train/Val only; Test was not accessed."

