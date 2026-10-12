#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-${VARIANT:-mm2_memory}}"
case "$MODE" in preflight|build_memory|smoke|screen|formal|train|evaluate|summarize|continue) ;;
  *) echo "Choose preflight|build_memory|smoke|screen|formal|train|evaluate|summarize|continue" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
HERE="$BDA_ROOT/experiments/spi_building_prototype_memory_v1"
SEED="${SEED:-3407}"
STEPS="${STEPS:-5000}"
TAG="${TAG:-probe${STEPS}}"
DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
TRAIN_LIST="${TRAIN_LIST:-$DATASET_ROOT/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-$DATASET_ROOT/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a0_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_PROTOTYPE_MEMORY_V1}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_BUILDING_PROTOTYPE_MEMORY_V1}"
MEMORY_PATH="${MEMORY_PATH:-$OUTPUT_ROOT/train_only_memory_seed${SEED}.pt}"
VARIANTS=(mm1_no_memory mm2_memory mm4_event_excluded mm5_shuffled_labels)

preflight() {
  "$PYTHON_BIN" -m py_compile \
    "$BDA_ROOT/model/DamageFormer_BuildingPrototypeMemory_V1.py" \
    "$HERE/core.py" "$HERE/build_memory.py" "$HERE/train.py" \
    "$HERE/evaluate.py" "$HERE/summarize.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "SPI Building Prototype Memory V1 preflight passed. Test access: FORBIDDEN."
}

inputs() {
  for path in "$PARENT_CKPT" "$DATASET_ROOT" "$TRAIN_LIST" "$VAL_LIST"; do
    [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  done
  [[ "$(basename "$TRAIN_LIST")" != *test* ]] || { echo "Refusing test list" >&2; exit 1; }
  [[ "$(basename "$VAL_LIST")" != *test* ]] || { echo "Refusing test list" >&2; exit 1; }
}

build_memory() {
  mkdir -p "$OUTPUT_ROOT"
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_building_prototype_memory_v1.build_memory \
    --dataset_path "$DATASET_ROOT" --train_list "$TRAIN_LIST" \
    --parent_ckpt "$PARENT_CKPT" --output "$MEMORY_PATH" \
    --purity "${PURITY:-0.8}" --batch_size "${MEMORY_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}" --seed "$SEED"
}

run_dir() { echo "$OUTPUT_ROOT/${1}_${TAG}_seed${SEED}"; }
result_file() { echo "$RESULT_ROOT/${1}_${TAG}.json"; }

train_one() {
  local variant="$1" run resume_args=()
  run="$(run_dir "$variant")"
  mkdir -p "$run"
  if [[ -s "$run/training_summary.json" ]]; then
    echo "Preserving completed training: $run"
    return
  fi
  [[ -s "$run/latest_resume.pth" ]] && resume_args=(--resume "$run/latest_resume.pth")
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_building_prototype_memory_v1.train \
    --variant "$variant" --dataset_path "$DATASET_ROOT" \
    --train_list "$TRAIN_LIST" \
    --parent_ckpt "$PARENT_CKPT" --memory_path "$MEMORY_PATH" \
    --output_dir "$run" --optimizer_steps "$STEPS" \
    --batch_size "${BATCH_SIZE:-4}" --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
    --crop_size "${CROP_SIZE:-640}" --num_workers "${NUM_WORKERS:-8}" \
    --lr "${LEARNING_RATE:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --false_positive_weight "${FALSE_POSITIVE_WEIGHT:-0.25}" \
    --nonharm_weight "${NONHARM_WEIGHT:-0.1}" --save_every "${SAVE_EVERY:-500}" \
    --device "${DEVICE:-cuda:0}" --seed "$SEED" "${resume_args[@]}"
}

evaluate_one() {
  local variant="$1" checkpoint="" output
  output="$(result_file "$variant")"
  mkdir -p "$RESULT_ROOT"
  if [[ "$variant" != mm0_frozen_a0 ]]; then
    checkpoint="$(run_dir "$variant")/final.pth"
    [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  fi
  local checkpoint_args=()
  [[ -n "$checkpoint" ]] && checkpoint_args=(--checkpoint "$checkpoint")
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_building_prototype_memory_v1.evaluate \
    --variant "$variant" "${checkpoint_args[@]}" \
    --parent_ckpt "$PARENT_CKPT" --memory_path "$MEMORY_PATH" \
    --dataset_path "$DATASET_ROOT" --val_list "$VAL_LIST" --output_json "$output" \
    --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}"
}

summarize_all() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_building_prototype_memory_v1.summarize \
    --root "$RESULT_ROOT" --tag "$TAG" \
    --output "$RESULT_ROOT/MM0_MM5_${TAG}_admission.json"
}

chain() {
  build_memory
  for variant in "${VARIANTS[@]}"; do train_one "$variant"; done
  evaluate_one mm0_frozen_a0
  for variant in "${VARIANTS[@]}"; do evaluate_one "$variant"; done
  summarize_all
}

screen_train_only() {
  build_memory
  for variant in "${VARIANTS[@]}"; do train_one "$variant"; done
  echo "500-step screen complete. Validation/Test were not accessed. Inspect training loss and gain before formal training."
}

[[ "$MODE" == preflight ]] && { preflight; exit 0; }
inputs
case "$MODE" in
  build_memory) preflight; build_memory ;;
  smoke) preflight; STEPS=2; TAG=smoke2; build_memory; train_one "$VARIANT" ;;
  screen) preflight; STEPS=500; TAG=screen500; screen_train_only ;;
  formal) preflight; STEPS=5000; TAG=probe5000; chain ;;
  train) preflight; build_memory; train_one "$VARIANT" ;;
  evaluate) preflight; build_memory; evaluate_one "$VARIANT" ;;
  summarize) preflight; summarize_all ;;
  continue) preflight; chain ;;
esac
