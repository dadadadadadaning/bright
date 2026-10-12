#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|pilot|evaluate|summarize|screen|formal|auto|continue) ;; *) echo "Choose preflight|smoke|pilot|evaluate|summarize|screen|formal|auto|continue" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"; PROJECT_ROOT="$(dirname "$BDA_ROOT")"; export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_superpixel_relation_occurrence_v1"; PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
SEED="${SEED:-7341}"; STEPS="${STEPS:-5000}"; P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"; SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_SUPERPIXEL_RELATION_OCCURRENCE_V2}"; RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_SUPERPIXEL_RELATION_OCCURRENCE_V2}"
preflight() { local protocol; protocol="$("$PYTHON_BIN" -c 'from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1 import PROTOCOL; print(PROTOCOL)')"; test "$protocol" = "SPI_SUPERPIXEL_RELATION_OCCURRENCE_V2" || { echo "Mixed experiment files: run.sh expects V2 but Python reports $protocol" >&2; exit 1; }; "$PYTHON_BIN" -c "import skimage; print('scikit-image', skimage.__version__)"; "$PYTHON_BIN" -m py_compile "$BDA_ROOT/model/DamageFormer_SuperpixelRelationOccurrence_V1.py" "$HERE/core.py" "$HERE/train.py" "$HERE/evaluate.py" "$HERE/summarize.py"; "$PYTHON_BIN" -m pytest -q "$BDA_ROOT/model/test_superpixel_relation_occurrence_v1.py"; bash -n "$HERE/run.sh"; echo "Protocol $protocol"; echo "Superpixel Relation Occurrence V2 preflight passed"; }
# The split lists are the authoritative inputs.  The manifest is convenient metadata,
# but evaluation must remain recoverable if that derived JSON was removed.
inputs() { for p in "$P3_CKPT" "$DATASET_ROOT" "$SPLIT_ROOT/meta_train.txt" "$SPLIT_ROOT/meta_calibration.txt"; do test -e "$p" || { echo "Missing input: $p" >&2; exit 1; }; done; }
count_calibration() { wc -l < "$SPLIT_ROOT/meta_calibration.txt" | tr -d '[:space:]'; }
train_one() {
  local v="$1"
  local run="$OUTPUT_ROOT/${v}_step${STEPS}_seed${SEED}"
  mkdir -p "$run"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.train --variant "$v" --base-checkpoint "$P3_CKPT" --dataset-path "$DATASET_ROOT" --train-list "$SPLIT_ROOT/meta_train.txt" --output-dir "$run" --device "${DEVICE:-cuda:0}" --seed "$SEED" --crop-size "${CROP_SIZE:-640}" --batch-size "${BATCH_SIZE:-2}" --accumulation-steps "${ACCUMULATION_STEPS:-8}" --optimizer-steps "$STEPS" --num-workers "${NUM_WORKERS:-4}" --learning-rate "${LEARNING_RATE:-1e-4}" --segments "${SEGMENTS:-192}" --compactness "${COMPACTNESS:-8}" --shift-cap "${SHIFT_CAP:-4.5}" --lambda-margin "${LAMBDA_MARGIN:-0.25}" --lambda-consistency "${LAMBDA_CONSISTENCY:-0.02}" --min-building-fraction "${MIN_BUILDING_FRACTION:-0.25}" --max-consecutive-skips "${MAX_CONSECUTIVE_SKIPS:-500}"
}
evaluate_one() {
  local v="$1"
  local run="$OUTPUT_ROOT/${v}_step${STEPS}_seed${SEED}"
  test -s "$run/last.pth" || { echo "Missing checkpoint $run/last.pth" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.evaluate --checkpoint "$run/last.pth" --dataset-path "$DATASET_ROOT" --list-path "$SPLIT_ROOT/meta_calibration.txt" --output "$RESULT_ROOT/${v}_step${STEPS}_seed${SEED}.json" --expected-images "$(count_calibration)" --device "${DEVICE:-cuda:0}" --num-workers "${EVAL_WORKERS:-2}" --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-2000}" --seed "$SEED"
}
summarize_all() { "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.summarize --result-root "$RESULT_ROOT" --steps "$STEPS" --seed "$SEED" --output "$RESULT_ROOT/SP0_SP5_step${STEPS}_seed${SEED}_summary.json"; }
run_chain() { for v in sp1_parameter_matched sp2_slic_relation sp3_slic_consistency sp4_shuffled_slic sp5_regular_grid; do train_one "$v"; evaluate_one "$v"; done; summarize_all; }
continue_chain() {
  for v in sp1_parameter_matched sp2_slic_relation sp3_slic_consistency sp4_shuffled_slic sp5_regular_grid; do
    local run="$OUTPUT_ROOT/${v}_step${STEPS}_seed${SEED}"
    local result="$RESULT_ROOT/${v}_step${STEPS}_seed${SEED}.json"
    if [[ ! -s "$run/last.pth" ]]; then
      echo "[continue] training missing checkpoint: $v"
      train_one "$v"
    else
      echo "[continue] preserving checkpoint: $v"
    fi
    if [[ ! -s "$result" ]]; then
      echo "[continue] evaluating: $v"
      evaluate_one "$v"
    else
      echo "[continue] preserving evaluation: $v"
    fi
  done
  summarize_all
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }; inputs
case "$MODE" in smoke) preflight; STEPS=2; for v in sp1_parameter_matched sp2_slic_relation sp3_slic_consistency sp4_shuffled_slic sp5_regular_grid; do train_one "$v"; done; echo "All SP1-SP5 smoke tests passed; meta-calibration/Val450/Test: NEVER." ;; pilot) preflight; train_one "${VARIANT:-sp2_slic_relation}" ;; evaluate) preflight; evaluate_one "${VARIANT:-sp2_slic_relation}" ;; summarize) preflight; summarize_all ;; screen) preflight; STEPS=500; run_chain; echo "SPR V2 500-step screen complete. This is diagnostic evidence, not a final claim." ;; formal) preflight; STEPS=5000; run_chain; echo "SPR V2 5000-step formal chain complete. Meta-confirm/Val450/Test: NEVER." ;; auto) preflight; run_chain; echo "SPR V2 custom-step chain complete: STEPS=$STEPS. Meta-confirm/Val450/Test: NEVER." ;; continue) preflight; continue_chain; echo "SPR V2 continuation complete: STEPS=$STEPS. Meta-confirm/Val450/Test: NEVER." ;; esac
