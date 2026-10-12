#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDA_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$BDA_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SEED="${SEED:-7341}"
CACHE_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_PI_HIERARCHICAL_MEMORY_V1/seed${SEED}}"

case "${1:-auto}" in
  preflight)
    "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
    "$PYTHON_BIN" -m py_compile "$BDA_ROOT/model/PseudoInstanceHierarchicalMemory.py" "$HERE/train_memory.py"
    echo "PI Hierarchical Memory preflight passed"
    ;;
  train|auto)
    if [[ "${1:-auto}" == "auto" ]]; then
      # Synced shell files do not always retain their executable bit.  Invoke
      # the nested preflight through bash so `bash run.sh auto` is sufficient.
      bash "$0" preflight
    fi
    "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_pseudo_instance_hierarchical_memory_v1.train_memory \
      --train-cache "$CACHE_ROOT/meta_train.pt" \
      --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
      --output-dir "$OUTPUT_ROOT" \
      --groups "${MEMORY_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}" \
      --hidden "${MEMORY_HIDDEN:-128}" \
      --capacity-per-class "${MEMORY_CAPACITY_PER_CLASS:-2048}" \
      --topk "${MEMORY_TOPK:-4}" \
      --parametric-steps "${PARAMETRIC_STEPS:-1000}" \
      --retrieval-steps "${RETRIEVAL_STEPS:-300}" \
      --batch-size "${BATCH_SIZE:-1024}" \
      --learning-rate "${LEARNING_RATE:-1e-3}" \
      --seed "$SEED"
    ;;
  *) echo "usage: $0 {preflight|train|auto}" >&2; exit 2 ;;
esac
