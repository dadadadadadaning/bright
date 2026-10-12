#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
EXP="${EXP:-experiments.damageformer_evidence_relation_memory_v1}"

TRAIN_CACHE="${TRAIN_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_train_all_buildings.pt}"
VAL_CACHE="${VAL_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_val_all_buildings.pt}"

OUT_ROOT="${OUT_ROOT:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_EVIDENCE_RELATION_MEMORY_V1}"
EVAL_ROOT="${EVAL_ROOT:-/root/data/BRIGHT-eval-results/DAMAGEFORMER_EVIDENCE_RELATION_MEMORY_V1}"

SPEC_DIR="$OUT_ROOT/specialist_c1"
SPEC_TRAIN="$SPEC_DIR/train_scores.pt"
SPEC_VAL="$SPEC_DIR/val_scores.pt"

mkdir -p "$OUT_ROOT" "$EVAL_ROOT"
cd "$ROOT"
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

MODE="${1:-help}"
CONTROL="${2:-none}"
HISTORY="$OUT_ROOT/history_${CONTROL}.pt"

case "$MODE" in
  preflight)
    python - <<'PY'
import sys, torch, numpy
print("impl =",sys.implementation.name)
print("python =",sys.version)
print("torch =",torch.__version__)
print("numpy =",numpy.__version__)
print("cuda =",torch.cuda.is_available())
assert sys.implementation.name=="cpython"
assert torch.cuda.is_available()
PY
    python -m "${EXP}.test_contract"
    python - <<PY
from ${EXP}.cache import load_stage1_cache
tr=load_stage1_cache("$TRAIN_CACHE","train")
va=load_stage1_cache("$VAL_CACHE","val")
print("train buildings =",len(tr["labels"]),"regions =",len(tr["region_tokens"]))
print("val buildings   =",len(va["labels"]),"regions =",len(va["region_tokens"]))
print("val base-intact =",int((va["base_pred"]==0).sum()))
print("test_accessed   =",tr["test_accessed"],va["test_accessed"])
PY
    ;;

  specialist)
    python -m "${EXP}.specialist" \
      --train-cache "$TRAIN_CACHE" \
      --val-cache "$VAL_CACHE" \
      --output-dir "$SPEC_DIR"
    ;;

  history)
    test -f "$SPEC_TRAIN" || { echo "Missing $SPEC_TRAIN; run specialist first"; exit 2; }
    python -m "${EXP}.history" \
      --train-cache "$TRAIN_CACHE" \
      --val-cache "$VAL_CACHE" \
      --specialist-train-scores "$SPEC_TRAIN" \
      --output "$HISTORY" \
      --control "$CONTROL"
    ;;

  relation)
    HISTORY="$OUT_ROOT/history_none.pt"
    test -f "$HISTORY" || { echo "Missing $HISTORY; run history none first"; exit 2; }
    python -m "${EXP}.relation_audit" \
      --train-cache "$TRAIN_CACHE" \
      --val-cache "$VAL_CACHE" \
      --specialist-train-scores "$SPEC_TRAIN" \
      --specialist-val-scores "$SPEC_VAL" \
      --history "$HISTORY" \
      --output "$EVAL_ROOT/E2_1_relation_audit.json"
    ;;

  e3)
    test -f "$HISTORY" || { echo "Missing $HISTORY; run history $CONTROL first"; exit 2; }
    python -m "${EXP}.evaluate_history" \
      --val-cache "$VAL_CACHE" \
      --history "$HISTORY" \
      --output "$EVAL_ROOT/E3_${CONTROL}.json"
    ;;

  utility)
    HISTORY="$OUT_ROOT/history_none.pt"
    test -f "$HISTORY" || { echo "Missing $HISTORY"; exit 2; }
    python -m "${EXP}.train_utility" \
      --train-cache "$TRAIN_CACHE" \
      --val-cache "$VAL_CACHE" \
      --specialist-train-scores "$SPEC_TRAIN" \
      --specialist-val-scores "$SPEC_VAL" \
      --history "$HISTORY" \
      --output-dir "$OUT_ROOT/utility"
    ;;

  e4)
    python -m "${EXP}.evaluate_utility" \
      --val-cache "$VAL_CACHE" \
      --utility-scores "$OUT_ROOT/utility/val_utility_scores.pt" \
      --output "$EVAL_ROOT/E4_utility.json"
    ;;

  summarize)
    python -m "${EXP}.summarize" --eval-root "$EVAL_ROOT"
    ;;

  all-e3)
    bash "$0" preflight
    bash "$0" specialist
    bash "$0" history none
    bash "$0" relation
    bash "$0" e3 none
    ;;

  controls)
    # Expensive: each local-memory control reruns Region matching.
    for C in shuffle_labels shuffle_events wrong_regions swapped_id; do
      bash "$0" history "$C"
      bash "$0" e3 "$C"
    done
    ;;

  *)
    cat <<EOF
Usage:
  bash run.sh preflight
  bash run.sh specialist
  bash run.sh history none
  bash run.sh relation
  bash run.sh e3 none
  bash run.sh controls
  bash run.sh utility      # ONLY after E3 real memory passes causal checks
  bash run.sh e4
  bash run.sh summarize
  bash run.sh all-e3

Protocol:
  Val coverage fixed at 2%.
  Test access is forbidden.
  E3 primary comparisons:
    Prototype -> Global Memory -> Local Region Memory -> Weighted Local Memory.
EOF
    ;;
esac
