#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|build|e0|e1|verify|auto|e2) ;; *) echo "Choose preflight|smoke|build|e0|e1|verify|auto|e2" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"; export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_local_structural_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Local Structural Semantic Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/build_local_cache.py" "$HERE/audit_e0.py" "$HERE/train_e1.py" "$HERE/verify_results.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "LSM preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_LOCAL_STRUCTURAL_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_LOCAL_STRUCTURAL_MEMORY_V1}"
TRAIN="$CACHE_ROOT/meta_train_local.pt"; CAL="$CACHE_ROOT/meta_calibration_local.pt"
E0="$RESULT_ROOT/lsm_e0_local_retrieval_seed${SEED}.json"; E1="$RESULT_ROOT/lsm_e1_dependency_seed${SEED}.json"
for path in "$DATASET_ROOT" "$P3_CKPT" "$SAR_STATS_JSON" "$SPLIT_ROOT/split_manifest.json"; do test -e "$path" || { echo "Missing input: $path" >&2; exit 1; }; done
mkdir -p "$CACHE_ROOT" "$RESULT_ROOT"
TRAIN_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_calibration'])")"

build_one() {
  local split="$1" count="$2" maximum="$3" output="$4"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_local_structural_memory_v1.build_local_cache \
    --checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/${split}.txt" --sar-stats-json "$SAR_STATS_JSON" \
    --output "$output" --split "$split" --expected-images "$count" --max-images "$maximum" \
    --segments "${STRUCTURAL_SEGMENTS:-192}" --compactness "${STRUCTURAL_COMPACTNESS:-8}" --minimum-area "${MINIMUM_AREA:-32}" \
    --images-per-shard "${IMAGES_PER_SHARD:-10}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" --progress-every "${PROGRESS_EVERY:-10}" --seed "$SEED"
}
build_all() { [[ -s "$TRAIN" ]] || build_one meta_train "$TRAIN_COUNT" 0 "$TRAIN"; [[ -s "$CAL" ]] || build_one meta_calibration "$CAL_COUNT" 0 "$CAL"; }
e0() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_local_structural_memory_v1.audit_e0 --train-cache "$TRAIN" --calibration-cache "$CAL" --output "$E0" \
    --topk "${MEMORY_TOPK:-8}" --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}" --minimum-purity "${MINIMUM_PURITY:-0.60}" --device "${DEVICE:-cuda:0}" --seed "$SEED"
}
e1() {
  test -s "$E0" || { echo "Run E0 first: missing $E0" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_local_structural_memory_v1.train_e1 --train-cache "$TRAIN" --calibration-cache "$CAL" --e0-result "$E0" --output "$E1" --checkpoint-dir "$CACHE_ROOT/e1_heads" \
    --topk "${MEMORY_TOPK:-8}" --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}" --minimum-purity "${MINIMUM_PURITY:-0.60}" \
    --maximum-train-queries-per-event-class "${MAXIMUM_TRAIN_QUERIES_PER_EVENT_CLASS:-512}" \
    --steps "${E1_STEPS:-600}" --batch-size "${E1_BATCH_SIZE:-512}" --hidden "${MEMORY_HIDDEN:-128}" --learning-rate "${LEARNING_RATE:-0.001}" --device "${DEVICE:-cuda:0}" --seed "$SEED"
}
case "$MODE" in
  smoke) build_one meta_train "$TRAIN_COUNT" "${SMOKE_IMAGES:-10}" "$CACHE_ROOT/smoke_local.pt" ;;
  build) build_all ;;
  e0) test -s "$TRAIN"; test -s "$CAL"; e0 ;;
  e1) test -s "$TRAIN"; test -s "$CAL"; e1 ;;
  verify) "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_local_structural_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED" ;;
  auto) preflight; build_all; e0; "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_local_structural_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED" ;;
  e2) echo "LSM-E2 is deliberately locked. It requires an explicit E1 admission and a new frozen protocol; no pixel writeback is implemented here." >&2; exit 3 ;;
esac
