#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
MOD="bda_benchmark.experiments.damageformer_late_decision_memory_v2"
cd "$ROOT"
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}
MODE="${1:-help}"
case "$MODE" in
  preflight)
    python - <<'PY'
import sys,torch,numpy
print("impl =",sys.implementation.name); print("python =",sys.version); print("torch =",torch.__version__); print("numpy =",numpy.__version__); print("cuda =",torch.cuda.is_available())
assert sys.implementation.name=="cpython"; assert torch.cuda.is_available()
PY
    python -m "${MOD}.test_core" ;;
  inspect-cache)
    CACHE="${2:?run.sh inspect-cache CACHE.pt}"; python -m "${MOD}.build_memory" --source "$CACHE" --inspect ;;
  build-memory)
    SRC="${2:?run.sh build-memory canonical.pt memory.pt}"; OUT="${3:?run.sh build-memory canonical.pt memory.pt}"; python -m "${MOD}.build_memory" --source "$SRC" --output "$OUT" --half ;;
  train-utility)
    CAL="${2:?run.sh train-utility train_cal.pt utility.pth}"; OUT="${3:?run.sh train-utility train_cal.pt utility.pth}"; python -m "${MOD}.train_utility" --train-cal "$CAL" --output "$OUT" ;;
  evaluate)
    IN="${2:?run.sh evaluate evaluator_input.pt summary.json}"; OUT="${3:?run.sh evaluate evaluator_input.pt summary.json}"; python -m "${MOD}.evaluate" --input "$IN" --output "$OUT" ;;
  *) echo "Modes: preflight | inspect-cache | build-memory | train-utility | evaluate" ;;
esac
