#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_qmetric_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$MODULE.test_partition" "$MODULE.test_metric" "$MODULE.test_joint" "$MODULE.test_pipeline" -v
  for entry in train_front train_metric train_joint report; do
    "$PYTHON" -m "$MODULE.$entry" --help >/dev/null
  done
  echo 'Local Q/metric/joint contracts passed. Real CUDA/data smoke is still required. No Test.'
  exit 0
fi
case "$MODE" in
  q|control|smoke_q|metric_joint|smoke_metric_joint|metric|joint) ;;
  *) echo 'Use preflight|smoke_q|q|control|smoke_metric_joint|metric_joint|metric|joint'; exit 2 ;;
esac

: "${EXPERIMENT_ROOT:?Set a NEW isolated EXPERIMENT_ROOT; do not use the running parent RUN_ROOT}"
: "${A1_CKPT:?Set the exact original A1 checkpoint}"
test -s "$A1_CKPT"
DATASET="${DATASET:-$PROJECT/bda_benchmark/dataset}"
TRAIN_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="$PROJECT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-3407}"
COMMON=(--dataset "$DATASET" --train_list "$TRAIN_LIST" --val_list "$VAL_LIST"
        --device "$DEVICE" --workers "$WORKERS")

if [[ "$MODE" == q || "$MODE" == control || "$MODE" == smoke_q ]]; then
  VARIANT=q_anneal
  EXTRA=()
  REFERENCE=()
  STEPS=50000
  EVAL=1000
  if [[ "$MODE" == control ]]; then VARIANT=control; fi
  if [[ "$MODE" == q ]]; then
    : "${CONTROL_FRONT:?Set the current parent/front directory to verify a fair Q control}"
    REFERENCE=(--reference_run "$CONTROL_FRONT")
  fi
  if [[ "$MODE" == smoke_q ]]; then
    STEPS=20
    EVAL=20
    # Smoke exercises BOTH beta=1 and beta=2 checkpoint restore, unlike a
    # 20-step smoke with the formal 1000-step hold period.
    EXTRA=(--smoke --q_beta_hold 2 --q_beta_ramp_end 10)
  fi
  "$PYTHON" -u -m "$MODULE.train_front" "${COMMON[@]}" "${EXTRA[@]}" "${REFERENCE[@]}" \
    --a1_checkpoint "$A1_CKPT" --output "$EXPERIMENT_ROOT/front" \
    --variant "$VARIANT" --steps "$STEPS" --eval_every "$EVAL" \
    --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --seed "$SEED" --amp "${AMP:-bf16}"
  echo "COMPLETE $MODE: $EXPERIMENT_ROOT/front. No Memory stage auto-launched."
  exit 0
fi

# This experiment intentionally uses the COMPLETED current parent front/bank.
# Q experiments have a different embedding and must not reuse this Memory bank.
: "${PARENT_RUN:?Set a completed damageformer_srac_transformer_v1 run (formal or matching smoke)}"
"$PYTHON" -c 'import pathlib,sys; out=pathlib.Path(sys.argv[1]).resolve(); parent=pathlib.Path(sys.argv[2]).resolve(); assert out != parent and parent not in out.parents, "EXPERIMENT_ROOT must be outside the parent result directory"' "$EXPERIMENT_ROOT" "$PARENT_RUN"
FRONT_CKPT="$PARENT_RUN/front/best.pth"
CACHE="$PARENT_RUN/train_memory"
test -s "$FRONT_CKPT"
test -s "$CACHE/manifest.json"
test -s "$CACHE/bank.pt"
EXTRA=()
METRIC_STEPS=2000
JOINT_STEPS=5000
JOINT_EVAL=500
if [[ "$MODE" == smoke_metric_joint ]]; then
  EXTRA=(--smoke)
  METRIC_STEPS=20
  JOINT_STEPS=20
  JOINT_EVAL=20
fi
METRIC_CKPT="${METRIC_CKPT:-$EXPERIMENT_ROOT/metric/metric_last.pth}"

metric() {
  "$PYTHON" -u -m "$MODULE.train_metric" "${EXTRA[@]}" \
    --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" --cache "$CACHE" \
    --output "$EXPERIMENT_ROOT/metric" --steps "$METRIC_STEPS" \
    --seed "$SEED" --device "$DEVICE"
}
joint() {
  "$PYTHON" -u -m "$MODULE.train_joint" "${COMMON[@]}" "${EXTRA[@]}" \
    --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" --cache "$CACHE" \
    --metric_checkpoint "$METRIC_CKPT" --output "$EXPERIMENT_ROOT/joint" \
    --steps "$JOINT_STEPS" --eval_every "$JOINT_EVAL" --topk 8 \
    --seed "$SEED" --amp "${AMP:-bf16}"
}
case "$MODE" in
  metric_joint|smoke_metric_joint)
    if [[ -e "$EXPERIMENT_ROOT" ]]; then
      echo "Output exists; preserve it and choose a new EXPERIMENT_ROOT: $EXPERIMENT_ROOT"; exit 2
    fi
    metric
    joint
    ;;
  *) "$MODE" ;;
esac
echo "COMPLETE $MODE: $EXPERIMENT_ROOT. Parent artifacts untouched. No Test."
