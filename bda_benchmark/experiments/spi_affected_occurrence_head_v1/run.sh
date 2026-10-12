#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|pilot|evaluate|auto) ;; *) echo "Choose preflight|smoke|pilot|evaluate|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"; export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_affected_occurrence_head_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
SEED="${SEED:-7341}"; STEPS="${STEPS:-500}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/data/BRIGHT-eval-results/SPI_P3_GATE_MISS_SPATIAL_AUDIT_V1}"
THRESHOLDS="$AUDIT_ROOT/p3_gate_miss_spatial_thresholds_seed${SEED}.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_AFFECTED_OCCURRENCE_HEAD_V1}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_AFFECTED_OCCURRENCE_HEAD_V1}"

preflight() {
  "$PYTHON_BIN" -m py_compile "$BDA_ROOT/model/DamageFormer_AffectedOccurrence_V1.py" "$HERE/spatial.py" "$HERE/train.py" "$HERE/evaluate.py"
  "$PYTHON_BIN" -m pytest -q "$BDA_ROOT/model/test_affected_occurrence_v1.py"
  bash -n "$HERE/run.sh"; echo "Affected Occurrence Head preflight passed"
}

inputs() {
  for p in "$P3_CKPT" "$DATASET_ROOT" "$SPLIT_ROOT/meta_train.txt" "$SPLIT_ROOT/meta_calibration.txt" "$THRESHOLDS"; do test -e "$p" || { echo "Missing input: $p" >&2; exit 1; }; done
}

small_area() { "$PYTHON_BIN" -c "import json; print(json.load(open('$THRESHOLDS'))['small_damage_area_q25'])"; }
count_calibration() { "$PYTHON_BIN" -c "import json; print(json.load(open('$SPLIT_ROOT/split_manifest.json'))['counts']['meta_calibration'])"; }

train_one() {
  local variant="$1" run="$OUTPUT_ROOT/${variant}_step${STEPS}_seed${SEED}"
  mkdir -p "$run"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_affected_occurrence_head_v1.train \
    --variant "$variant" --base-checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --train-list "$SPLIT_ROOT/meta_train.txt" --output-dir "$run" \
    --small-area "$(small_area)" --device "${DEVICE:-cuda:0}" --seed "$SEED" --crop-size "${CROP_SIZE:-640}" --batch-size "${BATCH_SIZE:-2}" \
    --accumulation-steps "${ACCUMULATION_STEPS:-8}" --optimizer-steps "$STEPS" --num-workers "${NUM_WORKERS:-4}" --learning-rate "${LEARNING_RATE:-1e-4}" \
    --boundary-weight "${BOUNDARY_WEIGHT:-1.0}" --small-weight "${SMALL_WEIGHT:-1.0}" --log-every "${LOG_EVERY:-20}" --save-every "${SAVE_EVERY:-100}"
}

evaluate_one() {
  local variant="$1" run="$OUTPUT_ROOT/${variant}_step${STEPS}_seed${SEED}"
  test -s "$run/last.pth" || { echo "Missing checkpoint: $run/last.pth" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_affected_occurrence_head_v1.evaluate \
    --checkpoint "$run/last.pth" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/meta_calibration.txt" --output "$RESULT_ROOT/${variant}_step${STEPS}_seed${SEED}.json" \
    --small-area "$(small_area)" --device "${DEVICE:-cuda:0}" --num-workers "${EVAL_WORKERS:-2}" --expected-images "$(count_calibration)" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-2000}" --seed "$SEED"
}

[[ "$MODE" == preflight ]] && { preflight; exit 0; }
inputs
case "$MODE" in
  smoke) preflight; STEPS=2; train_one m1_occurrence; echo "Smoke passed; no calibration, Val450, or Test was accessed." ;;
  pilot) preflight; train_one "${VARIANT:-m1_occurrence}" ;;
  evaluate) preflight; evaluate_one "${VARIANT:-m1_occurrence}" ;;
  auto)
    preflight
    for variant in m1_occurrence m2_boundary m3_small_damage m4_sar_quality; do
      train_one "$variant"; evaluate_one "$variant"
    done
    echo "AO-H ablation chain complete. Meta-confirm/Val450/Test accessed: NEVER."
    ;;
esac
