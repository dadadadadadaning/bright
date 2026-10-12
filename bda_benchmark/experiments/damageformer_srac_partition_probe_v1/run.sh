#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_partition_probe_v1
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_core" "$MODULE.test_entry" "$MODULE.test_numerics" "$MODULE.test_repeatability" -v
    "$PYTHON" -m "$MODULE.audit" --help
    "$PYTHON" -m "$MODULE.repeatability" --help
    "$PYTHON" -m "$MODULE.report" --help
    echo "Local contracts passed. Real snapshot/CUDA smoke is still required. No training started."
    exit 0 ;;
  report)
    : "${PROBE_OUTPUT:?Set the existing probe result directory}"
    "$PYTHON" -m "$MODULE.report" --root "$PROBE_OUTPUT"
    exit 0 ;;
  repeatability|smoke_prior|prior|smoke_iterations|iterations|smoke_boundary|boundary|smoke_scale|scale|smoke_all|all) ;;
  *) echo "Use preflight|repeatability|smoke_prior|prior|smoke_iterations|iterations|smoke_boundary|boundary|smoke_scale|scale|smoke_all|all|report"; exit 2 ;;
esac
: "${SNAPSHOT_CKPT:?Set stable front_snapshot.pth with adjacent snapshot.json, NOT live best.pth}"
: "${A1_CKPT:?Set the original audited A1 file}"
: "${PROBE_OUTPUT:?Set a NEW output directory; existing evidence is preserved}"
test -s "$SNAPSHOT_CKPT"
test -s "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"
test -s "$A1_CKPT"
if [[ -e "$PROBE_OUTPUT" ]]; then
  echo "Output exists; preserve it and choose a new PROBE_OUTPUT: $PROBE_OUTPUT"
  exit 2
fi
# This must be inherited before Python creates any CUDA/cuBLAS context.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"
case "$CUBLAS_WORKSPACE_CONFIG" in
  :4096:8|:16:8) ;;
  *) echo "Invalid CUBLAS_WORKSPACE_CONFIG; use :4096:8 or :16:8 in a new process"; exit 2 ;;
esac
COMMON=(
  --checkpoint "$SNAPSHOT_CKPT" --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
  --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
  --output "$PROBE_OUTPUT"
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" --seed "${SEED:-3407}"
)
if [[ "$MODE" == repeatability ]]; then
  "$PYTHON" -u -m "$MODULE.repeatability" "${COMMON[@]}" \
    --image "${PROBE_IMAGE:-ukraine-conflict_00000089}" --repeats "${PROBE_REPEATS:-10}"
  echo "COMPLETE repeatability. Single-image diagnostic only. No training/Memory/Test."
  exit 0
fi
SUITE="${MODE#smoke_}"
EXTRA=()
if [[ "$MODE" == smoke_* ]]; then
  EXTRA+=(--smoke)
else
  EXTRA+=(--start_index "${PROBE_START_INDEX:-0}" --max_images "${PROBE_MAX_IMAGES:-0}")
fi
"$PYTHON" -u -m "$MODULE.audit" "${COMMON[@]}" --suite "$SUITE" \
  --prior_strength "${PROBE_PRIOR_STRENGTH:-2.0}" --leakage "${PROBE_LEAKAGE:-0.1}" \
  "${EXTRA[@]}"
echo "COMPLETE $MODE. Read-only experiment. No training/Memory/Test."
