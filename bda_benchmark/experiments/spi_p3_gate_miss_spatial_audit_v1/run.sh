#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|fit|evaluate|auto) ;; *) echo "Choose preflight|smoke|fit|evaluate|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"; export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_p3_gate_miss_spatial_audit_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "P3 Gate-Miss Spatial Audit Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "P3 Gate-Miss Spatial Audit preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_P3_GATE_MISS_SPATIAL_AUDIT_V1}"
THRESHOLDS="$RESULT_ROOT/p3_gate_miss_spatial_thresholds_seed${SEED}.json"
EVALUATION="$RESULT_ROOT/p3_gate_miss_spatial_meta_calibration_seed${SEED}.json"
for path in "$DATASET_ROOT" "$P3_CKPT" "$SAR_STATS_JSON" "$SPLIT_ROOT/split_manifest.json"; do test -e "$path" || { echo "Missing input: $path" >&2; exit 1; }; done
mkdir -p "$RESULT_ROOT"
count_for() { "$PYTHON_BIN" -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['$1'])"; }
run_one() {
  local mode="$1" split="$2" maximum="$3" output="$4"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_p3_gate_miss_spatial_audit_v1.audit \
    --mode "$mode" --split "$split" --checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/$split.txt" --sar-stats-json "$SAR_STATS_JSON" \
    --expected-images "$(count_for "$split")" --thresholds "$THRESHOLDS" --output "$output" --max-images "$maximum" \
    --core-radius "${CORE_RADIUS:-2}" --alignment-radius "${ALIGNMENT_RADIUS:-2}" --sar-mad-window "${SAR_MAD_WINDOW:-5}" \
    --minimum-building-area "${MINIMUM_BUILDING_AREA:-32}" \
    --high-risk-quantile "${HIGH_RISK_QUANTILE:-0.75}" --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --progress-every "${PROGRESS_EVERY:-10}" --seed "$SEED"
}
case "$MODE" in
  smoke)
    preflight
    smoke_thresholds="$RESULT_ROOT/smoke_thresholds.json"
    THRESHOLDS="$smoke_thresholds"
    run_one fit meta_train "${SMOKE_IMAGES:-10}" "$smoke_thresholds"
    run_one evaluate meta_train "${SMOKE_IMAGES:-10}" "$RESULT_ROOT/smoke_evaluation.json"
    ;;
  fit) preflight; run_one fit meta_train 0 "$THRESHOLDS" ;;
  evaluate) preflight; test -s "$THRESHOLDS" || { echo "Run fit first: $THRESHOLDS" >&2; exit 1; }; run_one evaluate meta_calibration 0 "$EVALUATION" ;;
  auto) preflight; test -s "$THRESHOLDS" || run_one fit meta_train 0 "$THRESHOLDS"; run_one evaluate meta_calibration 0 "$EVALUATION" ;;
esac
