#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|build|audit|resume-e2|verify|auto) ;; *) echo "Choose preflight|smoke|build|audit|resume-e2|verify|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_independent_building_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Independent Building Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/build_cards.py" "$HERE/audit.py" "$HERE/resume_e2.py" "$HERE/verify_results.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "Independent Building Memory V1 preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OLD_CARD_ROOT="${OLD_CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_COMPLETE_BUILDING_MEMORY_V1_SHARDED/seed${SEED}}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_INDEPENDENT_BUILDING_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_INDEPENDENT_BUILDING_MEMORY_V1}"
TRAIN="$CACHE_ROOT/meta_train_cards.pt"; CAL="$CACHE_ROOT/meta_calibration_cards.pt"
OLD_TRAIN="$OLD_CARD_ROOT/meta_train_cards.pt"; OLD_CAL="$OLD_CARD_ROOT/meta_calibration_cards.pt"
for path in "$DATASET_ROOT" "$P3_CKPT" "$SAR_STATS_JSON" "$SPLIT_ROOT/split_manifest.json" "$OLD_TRAIN" "$OLD_CAL"; do
  test -e "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }
done
mkdir -p "$CACHE_ROOT" "$RESULT_ROOT"
TRAIN_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_train'])")"
CAL_COUNT="$($PYTHON_BIN -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_calibration'])")"

build_one() {
  local split="$1" count="$2" maximum="$3" output="$4"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_independent_building_memory_v1.build_cards \
    --checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/${split}.txt" \
    --sar-stats-json "$SAR_STATS_JSON" --output "$output" --split "$split" --expected-images "$count" \
    --max-images "$maximum" --minimum-area "${MINIMUM_AREA:-32}" --evidence-side "${EVIDENCE_SIDE:-256}" \
    --tokens-per-role "${TOKENS_PER_ROLE:-3}" --images-per-shard "${IMAGES_PER_SHARD:-25}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-1}" --progress-every "${PROGRESS_EVERY:-10}" --seed "$SEED"
}
build_all() { [[ -s "$TRAIN" ]] || build_one meta_train "$TRAIN_COUNT" 0 "$TRAIN"; [[ -s "$CAL" ]] || build_one meta_calibration "$CAL_COUNT" 0 "$CAL"; }

audit() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_independent_building_memory_v1.audit \
    --train-independent "$TRAIN" --calibration-independent "$CAL" --train-complete "$OLD_TRAIN" --calibration-complete "$OLD_CAL" \
    --output-root "$CACHE_ROOT/models" --result-root "$RESULT_ROOT" --device "${AUDIT_DEVICE:-cuda:0}" \
    --hidden "${MEMORY_HIDDEN:-128}" --steps "${METRIC_STEPS:-800}" --role-steps "${ROLE_STEPS:-400}" \
    --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" \
    --prototypes-per-class "${PROTOTYPES_PER_CLASS:-3}" --capacities "${MEMORY_CAPACITIES:-32,64,128,256,512,0}" \
    --topk 8 --maximum-per-event 2 --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
}

if [[ "$MODE" == smoke ]]; then
  preflight
  build_one meta_train "$TRAIN_COUNT" "${SMOKE_IMAGES:-10}" "$CACHE_ROOT/smoke_train.pt"
  echo "Smoke card construction passed; run full audit after full train/calibration cards exist."
  exit 0
fi
case "$MODE" in
  build) build_all ;;
  audit) test -s "$TRAIN"; test -s "$CAL"; audit ;;
  resume-e2) "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_independent_building_memory_v1.resume_e2 \
    --train-independent "$TRAIN" --calibration-independent "$CAL" \
    --model-checkpoint "$CACHE_ROOT/models/mi_e1_conditional_residual.pth" \
    --d0-json "$RESULT_ROOT/mi_d0_d2_independent_information.json" \
    --d1-json "$RESULT_ROOT/mi_d1_capacity_diversity.json" --result-root "$RESULT_ROOT" \
    --device "${AUDIT_DEVICE:-cuda:0}" --hidden "${MEMORY_HIDDEN:-128}" --role-steps "${ROLE_STEPS:-400}" \
    --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" --topk 8 --maximum-per-event 2 \
    --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED" ;;
  verify) "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_independent_building_memory_v1.verify_results --cache-root "$CACHE_ROOT" --result-root "$RESULT_ROOT" ;;
  auto) preflight; build_all; audit ;;
esac
