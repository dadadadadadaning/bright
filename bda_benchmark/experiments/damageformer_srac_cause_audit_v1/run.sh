#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_cause_audit_v1
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_core" "$MODULE.test_support" "$MODULE.test_audit" "$MODULE.test_freeze_probe" -v
    for entry in logs snapshot audit report freeze_probe; do "$PYTHON" -m "$MODULE.$entry" --help >/dev/null; done
    echo "Cause-audit CPU contracts passed. Real BRIGHT/CUDA smoke still required. No training launched."
    exit 0 ;;
  logs)
    : "${PARENT_RUN:?Set the parent formal run directory (containing front/)}"
    "$PYTHON" -m "$MODULE.logs" --front_run "$PARENT_RUN/front"
    exit 0 ;;
  snapshot)
    : "${PARENT_RUN:?Set the parent formal run directory}"
    : "${CAUSE_ROOT:?Set a NEW isolated diagnostic root}"
    "$PYTHON" -m "$MODULE.snapshot" --source "$PARENT_RUN/front/best.pth" --output "$CAUSE_ROOT/snapshot"
    exit 0 ;;
  report)
    : "${CAUSE_ROOT:?Set the diagnostic root}"
    "$PYTHON" -m "$MODULE.report" --root "${CAUSE_OUTPUT:-$CAUSE_ROOT/audit}"
    exit 0 ;;
  smoke_audit|audit|smoke_freeze|freeze_probe) ;;
  *) echo "Use preflight|logs|snapshot|smoke_audit|audit|report|smoke_freeze|freeze_probe"; exit 2 ;;
esac
: "${CAUSE_ROOT:?Set the root containing snapshot/front_snapshot.pth}"
: "${A1_CKPT:?Set the audited original A1 checkpoint path}"
CHECKPOINT="$CAUSE_ROOT/snapshot/front_snapshot.pth"
test -s "$CHECKPOINT"
test -s "$A1_CKPT"
COMMON=(--checkpoint "$CHECKPOINT" --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
  --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" --seed "${SEED:-3407}")
case "$MODE" in
  smoke_audit|audit)
    EXTRA=(); if [[ "$MODE" == smoke_audit ]]; then EXTRA=(--smoke); fi
    "$PYTHON" -u -m "$MODULE.audit" "${COMMON[@]}" "${EXTRA[@]}" \
      --power "${Q_POWER:-2}" --threshold "${B_THRESHOLD:-0.5}" \
      --output "${CAUSE_OUTPUT:-$CAUSE_ROOT/$MODE}"
    ;;
  smoke_freeze|freeze_probe)
    EXTRA=(); if [[ "$MODE" == smoke_freeze ]]; then EXTRA=(--smoke); fi
    "$PYTHON" -u -m "$MODULE.freeze_probe" "${COMMON[@]}" "${EXTRA[@]}" \
      --output "${CAUSE_OUTPUT:-$CAUSE_ROOT/freeze_probe}" \
      --steps "${PROBE_STEPS:-3000}" --eval_every "${PROBE_EVAL_EVERY:-500}" \
      --batch "${PROBE_BATCH:-2}" --accum "${PROBE_ACCUM:-8}" \
      --lr_readout "${PROBE_LR_READOUT:-1e-4}" --lr_fusion "${PROBE_LR_FUSION:-1e-5}" \
      --amp "${PROBE_AMP:-none}"
    ;;
esac
echo "COMPLETE $MODE. No other stage launched. Train/Val only."
