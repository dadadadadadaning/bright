#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
ROOT="${ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
EXP="bda_benchmark.experiments.damageformer_decision_chain_v2"
TRAIN_CACHE="${TRAIN_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_train_all_buildings.pt}"
VAL_CACHE="${VAL_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_val_all_buildings.pt}"
OUT_ROOT="${OUT_ROOT:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_DECISION_CHAIN_V2}"
EVAL_ROOT="${EVAL_ROOT:-/root/data/BRIGHT-eval-results/DAMAGEFORMER_DECISION_CHAIN_V2}"
LOG_ROOT="${LOG_ROOT:-/root/data/BRIGHT-train-logs/DAMAGEFORMER_DECISION_CHAIN_V2}"
SEED="${SEED:-3407}"
DEVICE="${DEVICE:-cuda:0}"

cd "$ROOT"
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}
mkdir -p "$OUT_ROOT/stage1" "$OUT_ROOT/stage2" "$OUT_ROOT/stage3" "$EVAL_ROOT" "$LOG_ROOT"

preflight() {
  python -m py_compile experiments/damageformer_decision_chain_v2/*.py
  python -m unittest "${EXP}.test_contract" -v
  test -s "$TRAIN_CACHE" || { echo "Missing Train cache: $TRAIN_CACHE" >&2; exit 3; }
  test -s "$VAL_CACHE" || { echo "Missing Val cache: $VAL_CACHE" >&2; exit 3; }
  [[ "$TRAIN_CACHE" != *test* && "$VAL_CACHE" != *test* ]] || { echo "Test access forbidden" >&2; exit 4; }
  python - <<PY
from ${EXP}.common import load_stage1_cache
tr=load_stage1_cache("$TRAIN_CACHE","train")
va=load_stage1_cache("$VAL_CACHE","val")
print("train_buildings=",len(tr["labels"]),"val_buildings=",len(va["labels"]))
print("test_accessed=",tr["test_accessed"],va["test_accessed"])
PY
  echo "Decision Chain V2 preflight passed. Test access: FORBIDDEN."
}

stage1() {
  preflight
  for variant in C0 C1 C2 C3; do
    destination="$OUT_ROOT/stage1/$variant"
    if [[ -s "$destination/summary.json" ]]; then
      echo "===== $variant EXISTS; SKIP ====="
      continue
    fi
    echo "===== $variant START $(date -Iseconds) ====="
    python -u -m "${EXP}.stage1_specialist" \
      --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --variant "$variant" --output_dir "$destination" \
      --steps "${SPECIALIST_STEPS:-3000}" --eval_every "${SPECIALIST_EVAL_EVERY:-250}" \
      --seed "$SEED" --device "$DEVICE"
    echo "===== $variant COMPLETE $(date -Iseconds) ====="
  done
  python -m "${EXP}.select_specialist" \
    --root "$OUT_ROOT/stage1" --output "$OUT_ROOT/stage1/selected.json"
}

selected_scores() {
  python - <<PY
import json
from pathlib import Path
x=json.loads(Path("$OUT_ROOT/stage1/selected.json").read_text())
print(x["selected"]["scores"])
PY
}

stage2() {
  test -s "$OUT_ROOT/stage1/selected.json" || { echo "Run stage1 first" >&2; exit 5; }
  scores="$(selected_scores)"
  if [[ -s "$OUT_ROOT/stage2/summary.json" ]]; then
    echo "===== STAGE2 EXISTS; SKIP ====="
    return
  fi
  echo "===== SPECIALIST+PROTOTYPE START $(date -Iseconds) ====="
  python -u -m "${EXP}.stage2_relation" \
    --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
    --specialist_scores "$scores" --output_dir "$OUT_ROOT/stage2" --seed "$SEED"
  echo "===== SPECIALIST+PROTOTYPE COMPLETE $(date -Iseconds) ====="
}

stage3() {
  test -s "$OUT_ROOT/stage2/evidence.pt" || { echo "Run stage2 first" >&2; exit 6; }
  if [[ -s "$OUT_ROOT/stage3/summary.json" ]]; then
    echo "===== STAGE3 EXISTS; SKIP ====="
    return
  fi
  echo "===== 5D EVENT-OOF UTILITY START $(date -Iseconds) ====="
  python -u -m "${EXP}.stage3_utility" \
    --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
    --evidence "$OUT_ROOT/stage2/evidence.pt" --output_dir "$OUT_ROOT/stage3" \
    --steps "${UTILITY_STEPS:-600}" --seed "$SEED" --device "$DEVICE"
  echo "===== 5D EVENT-OOF UTILITY COMPLETE $(date -Iseconds) ====="
}

summarize() {
  python -m "${EXP}.summarize" \
    --root "$OUT_ROOT" --output "$EVAL_ROOT/decision_chain_summary.json"
}

case "$MODE" in
  preflight) preflight ;;
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  summarize) summarize ;;
  all)
    stage1
    stage2
    stage3
    summarize
    echo "===== DECISION CHAIN ALL COMPLETE $(date -Iseconds) ====="
    ;;
  status)
    pgrep -af 'damageformer_decision_chain_v2' || true
    find "$OUT_ROOT" "$EVAL_ROOT" -type f \( -name '*.json' -o -name '*.pt' -o -name '*.pth' \) -print 2>/dev/null || true
    ;;
  *) echo "Usage: $0 {preflight|stage1|stage2|stage3|summarize|all|status}" >&2; exit 2 ;;
esac

