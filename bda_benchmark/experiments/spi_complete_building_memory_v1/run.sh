#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|build|audit|controls|writeback|e1|auto) ;; *) echo "Choose preflight|smoke|build|audit|controls|writeback|e1|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_complete_building_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Complete Building Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/build_cards.py" "$HERE/build_cards_sharded.py" \
    "$HERE/audit_suitability.py" "$HERE/audit_semantic_controls.py" "$HERE/evaluate_writeback.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "Complete Building Memory V1 preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
D1_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_COMPLETE_BUILDING_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_COMPLETE_BUILDING_MEMORY_V1}"
TRAIN="$CACHE_ROOT/meta_train_cards.pt"; CAL="$CACHE_ROOT/meta_calibration_cards.pt"
OUTPUT="$RESULT_ROOT/memory_task_suitability_r0_seed${SEED}.json"
CONTROL_OUTPUT="$RESULT_ROOT/memory_semantic_controls_c0_c7_seed${SEED}.json"
WRITEBACK_OUTPUT="$RESULT_ROOT/memory_pixel_writeback_e1_seed${SEED}.json"
for path in "$DATASET_ROOT" "$P3_CKPT" "$SAR_STATS_JSON" "$SPLIT_ROOT/split_manifest.json" \
  "$D1_ROOT/meta_train.pt" "$D1_ROOT/meta_calibration.pt"; do
  test -e "$path" || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$CACHE_ROOT" "$RESULT_ROOT"
TRAIN_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_calibration'])")"

build_one() {
  local split="$1" count="$2" maximum="$3" output="$4"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_complete_building_memory_v1.build_cards_sharded \
    --checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/${split}.txt" \
    --sar-stats-json "$SAR_STATS_JSON" --output "$output" --split "$split" --expected-images "$count" \
    --max-images "$maximum" --minimum-area "${MINIMUM_AREA:-32}" --tokens-per-role "${TOKENS_PER_ROLE:-2}" \
    --images-per-shard "${IMAGES_PER_SHARD:-25}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" --progress-every "${PROGRESS_EVERY:-10}" --seed "$SEED"
}

if [[ "$MODE" == smoke ]]; then
  build_one meta_train "$TRAIN_COUNT" "${SMOKE_IMAGES:-10}" "$CACHE_ROOT/smoke.pt"
  echo "Complete Building Memory smoke passed; calibration/confirm/Val/test not accessed."
  exit 0
fi
build_all() {
  [[ -s "$TRAIN" ]] || build_one meta_train "$TRAIN_COUNT" 0 "$TRAIN"
  [[ -s "$CAL" ]] || build_one meta_calibration "$CAL_COUNT" 0 "$CAL"
}
audit() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_complete_building_memory_v1.audit_suitability \
    --train-cards "$TRAIN" --calibration-cards "$CAL" \
    --train-d1 "$D1_ROOT/meta_train.pt" --calibration-d1 "$D1_ROOT/meta_calibration.pt" \
    --output "$OUTPUT" --device "${AUDIT_DEVICE:-cuda:0}" --topk "${MEMORY_TOPK:-8}" \
    --coarse-buildings "${COARSE_BUILDINGS:-32}" --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}" \
    --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --seed "$SEED"
}
controls() {
  test -s "$OUTPUT" || { echo "Run audit first: missing $OUTPUT" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_complete_building_memory_v1.audit_semantic_controls \
    --train-cards "$TRAIN" --calibration-cards "$CAL" --suitability-json "$OUTPUT" \
    --output "$CONTROL_OUTPUT" --device "${AUDIT_DEVICE:-cuda:0}" --topk "${MEMORY_TOPK:-8}" \
    --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}" \
    --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --seed "$SEED"
}
writeback() {
  test -s "$OUTPUT" || { echo "Run audit first: missing $OUTPUT" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_complete_building_memory_v1.evaluate_writeback \
    --checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" \
    --list-path "$SPLIT_ROOT/meta_calibration.txt" --sar-stats-json "$SAR_STATS_JSON" \
    --train-cards "$TRAIN" --calibration-cards "$CAL" --suitability-json "$OUTPUT" \
    --output "$WRITEBACK_OUTPUT" --expected-images "$CAL_COUNT" --minimum-area "${MINIMUM_AREA:-32}" \
    --beta 0.25 --topk 8 --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}" \
    --maximum-per-event 2 --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --progress-every "${PROGRESS_EVERY:-10}" --seed "$SEED"
}
case "$MODE" in
  build) build_all ;;
  audit) test -s "$TRAIN"; test -s "$CAL"; audit ;;
  controls) test -s "$TRAIN"; test -s "$CAL"; controls ;;
  writeback) test -s "$TRAIN"; test -s "$CAL"; writeback ;;
  e1) test -s "$TRAIN"; test -s "$CAL"; controls; writeback ;;
  auto) preflight; build_all; audit ;;
esac
