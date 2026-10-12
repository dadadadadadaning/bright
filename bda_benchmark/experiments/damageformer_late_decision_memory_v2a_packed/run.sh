#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
MOD="bda_benchmark.experiments.damageformer_late_decision_memory_v2a_packed"
cd "$ROOT"
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

TRAIN_CACHE="${TRAIN_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_train_all_buildings.pt}"
VAL_CACHE="${VAL_CACHE:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_val_all_buildings.pt}"

case "${1:-help}" in
  audit)
    python -m "${MOD}.audit" \
      --train-cache "$TRAIN_CACHE" \
      --val-cache "$VAL_CACHE"
    ;;
  *)
    echo "Usage: bash run.sh audit"
    ;;
esac
