#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_snapshot_joint123_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$MODULE.test_contracts" \
    bda_benchmark.experiments.damageformer_srac_joint123_v1.test_joint123 \
    bda_benchmark.experiments.damageformer_srac_joint123_v1.test_entry -v
  for entry in precheck build_memory train_metric train report; do
    "$PYTHON" -m "$MODULE.$entry" --help >/dev/null
  done
  echo 'Snapshot contracts passed. Original completed-parent gates unchanged. CUDA/data tests still required.'
  exit 0
fi
if [[ "$MODE" == report ]]; then
  : "${SNAP_J123_ROOT:?Set existing isolated run root}"
  "$PYTHON" -m "$MODULE.report" \
    --root "$SNAP_J123_ROOT/joint"
  exit 0
fi
case "$MODE" in
  check|smoke_memory|memory|metric|probe|train|pipeline) ;;
  *) echo 'Use preflight|check|smoke_memory|memory|metric|probe|train|pipeline|report'; exit 2 ;;
esac
: "${SNAPSHOT_CKPT:?Set immutable front_snapshot.pth with snapshot.json}"
: "${A1_CKPT:?Set the original audited A1 weights}"
: "${SNAP_J123_ROOT:?Set NEW isolated run root, not a parent or snapshot directory}"
test -s "$SNAPSHOT_CKPT"
test -s "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"
test -s "$A1_CKPT"
DATASET="${DATASET:-$PROJECT/bda_benchmark/dataset}"
TRAIN_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-3407}"
SNAP_J123_AMP="${SNAP_J123_AMP:-none}"
CACHE="$SNAP_J123_ROOT/train_memory"
METRIC="$SNAP_J123_ROOT/metric/metric_last.pth"
# Never inherit PARENT_RUN, VARIANTS, OUTPUT_ROOT, BATCH or ACCUM.
"$PYTHON" -c 'import pathlib,sys; out=pathlib.Path(sys.argv[1]).resolve(); snap=pathlib.Path(sys.argv[2]).resolve().parent; assert out != snap and snap not in out.parents and out not in snap.parents, "Run root must be disjoint from snapshot directory"' "$SNAP_J123_ROOT" "$SNAPSHOT_CKPT"
CHECK=(--checkpoint "$SNAPSHOT_CKPT" --a1_checkpoint "$A1_CKPT"
       --train_list "$TRAIN_LIST" --val_list "$VAL_LIST")
COMMON=("${CHECK[@]}" --dataset "$DATASET" --workers "$WORKERS" --device "$DEVICE" --seed "$SEED")
check() { "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}"; }
smoke_memory() {
  check
  "$PYTHON" -u -m "$MODULE.build_memory" "${COMMON[@]}" \
    --output "$SNAP_J123_ROOT/smoke_memory" --smoke --per_event_class 64 --min_purity 0.0
}
memory() {
  check
  "$PYTHON" -u -m "$MODULE.build_memory" "${COMMON[@]}" \
    --output "$CACHE" --per_event_class 64 --min_purity 0.0
}
metric() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --cache "$CACHE"
  "$PYTHON" -u -m "$MODULE.train_metric" \
    --checkpoint "$SNAPSHOT_CKPT" --a1_checkpoint "$A1_CKPT" --cache "$CACHE" \
    --output "$SNAP_J123_ROOT/metric" --steps 2000 --seed "$SEED" --device "$DEVICE"
}
probe() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --cache "$CACHE" --metric_checkpoint "$METRIC"
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --cache "$CACHE" --metric_checkpoint "$METRIC" \
    --output "$SNAP_J123_ROOT/probe" --probe --steps 20 --eval_every 20 --amp "$SNAP_J123_AMP" --topk 8
}
train() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --cache "$CACHE" \
    --metric_checkpoint "$METRIC" --probe_root "$SNAP_J123_ROOT/probe"
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --cache "$CACHE" --metric_checkpoint "$METRIC" \
    --output "$SNAP_J123_ROOT/joint" --steps 5000 --eval_every 500 --amp "$SNAP_J123_AMP" --topk 8
}
case "$MODE" in
  pipeline)
    if [[ -e "$SNAP_J123_ROOT" ]]; then
      echo "Output exists; preserve it and select a NEW SNAP_J123_ROOT: $SNAP_J123_ROOT" >&2
      exit 2
    fi
    smoke_memory
    memory
    metric
    probe
    train
    ;;
  *) "$MODE" ;;
esac
echo "COMPLETE $MODE. Snapshot protocol ONLY; parent completion unchanged; Train/Val only. $SNAP_J123_ROOT"
