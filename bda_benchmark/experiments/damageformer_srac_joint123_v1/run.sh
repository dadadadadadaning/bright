#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_joint123_v1
METRIC_MODULE=bda_benchmark.experiments.damageformer_srac_qmetric_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$MODULE.test_joint123" "$MODULE.test_entry" -v
  for entry in train precheck report; do
    "$PYTHON" -m "$MODULE.$entry" --help >/dev/null
  done
  echo 'J1/J2/J3 local contracts passed. No Q changes. Real-data probe is still required.'
  exit 0
fi
case "$MODE" in
  check|prepare|probe|train|pipeline) ;;
  *) echo 'Use preflight|check|prepare|probe|train|pipeline'; exit 2 ;;
esac
: "${PARENT_RUN:?Set completed transformer_v1 parent run containing front/best.pth and train_memory}"
: "${A1_CKPT:?Set the exact original A1 checkpoint file}"
DATASET="${DATASET:-$PROJECT/bda_benchmark/dataset}"
TRAIN_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
CHECK=(--parent_run "$PARENT_RUN" --a1_checkpoint "$A1_CKPT"
       --train_list "$TRAIN_LIST" --val_list "$VAL_LIST")
if [[ "$MODE" == check ]]; then
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}"
  exit 0
fi
: "${J123_ROOT:?Set a NEW isolated J123_ROOT, not a parent result directory}"
"$PYTHON" -c 'import pathlib,sys; out=pathlib.Path(sys.argv[1]).resolve(); parent=pathlib.Path(sys.argv[2]).resolve(); assert out != parent and parent not in out.parents, "J123_ROOT must be outside PARENT_RUN"' "$J123_ROOT" "$PARENT_RUN"
FRONT_CKPT="$PARENT_RUN/front/best.pth"
CACHE="$PARENT_RUN/train_memory"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-3407}"
AMP="${AMP:-bf16}"
# Deliberately do not inherit generic VARIANTS, BATCH, ACCUM or METRIC_CKPT.
# Existing shells may retain incompatible values from previous experiments.
JOINT_METRIC="${J123_METRIC_CKPT:-$J123_ROOT/metric/metric_last.pth}"
COMMON=(--checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" --cache "$CACHE"
        --metric_checkpoint "$JOINT_METRIC" --dataset "$DATASET"
        --train_list "$TRAIN_LIST" --val_list "$VAL_LIST" --device "$DEVICE"
        --workers "$WORKERS" --seed "$SEED" --amp "$AMP" --topk 8)

prepare() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}"
  if [[ -n "${J123_METRIC_CKPT:-}" ]]; then
    "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --metric_checkpoint "$JOINT_METRIC"
    echo "Reusing the verified completed metric: $JOINT_METRIC"
  else
    "$PYTHON" -u -m "$METRIC_MODULE.train_metric" \
      --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" --cache "$CACHE" \
      --output "$J123_ROOT/metric" --steps 2000 --seed "$SEED" --device "$DEVICE"
  fi
}
probe() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --metric_checkpoint "$JOINT_METRIC"
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" \
    --output "$J123_ROOT/probe" --probe --steps 20 --eval_every 20
}
train() {
  "$PYTHON" -u -m "$MODULE.precheck" "${CHECK[@]}" --metric_checkpoint "$JOINT_METRIC" \
    --probe_root "$J123_ROOT/probe"
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" \
    --output "$J123_ROOT/joint" --steps 5000 --eval_every 500
}
case "$MODE" in
  pipeline)
    if [[ -e "$J123_ROOT" ]]; then
      echo "Output exists; choose a new J123_ROOT and preserve: $J123_ROOT"; exit 2
    fi
    prepare
    probe
    train
    ;;
  *) "$MODE" ;;
esac
echo "COMPLETE $MODE. Only J1/J2/J3. Q unchanged. No Test. Results: $J123_ROOT"
