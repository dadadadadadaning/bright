#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
ARG="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
SEED="${SEED:-3407}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset}"
TRAIN_LIST="${TRAIN_LIST:-$DATASET_PATH/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-$DATASET_PATH/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-$STORAGE_ROOT/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-$STORAGE_ROOT/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/p_sc_residual_seed3407/stage1_change/latest.pth}"
OUT_ROOT="${LATE_OUT_ROOT:-$STORAGE_ROOT/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1}"
EVAL_ROOT="${LATE_EVAL_ROOT:-$STORAGE_ROOT/BRIGHT-eval-results/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1}"
CACHE_ROOT="$OUT_ROOT/cache/p_sc_residual_seed${SEED}"
TRAIN_CACHE="$CACHE_ROOT/stage1_train_all_buildings.pt"
VAL_CACHE="$CACHE_ROOT/stage1_val_all_buildings.pt"
SPEC_ROOT="$OUT_ROOT/specialist"
mkdir -p "$CACHE_ROOT" "$SPEC_ROOT" "$EVAL_ROOT"

py() { python -u -m "bda_benchmark.experiments.damageformer_late_decision_correction_v1.$1" "${@:2}"; }

case "$MODE" in
  preflight)
    python -m py_compile experiments/damageformer_late_decision_correction_v1/*.py
    test -s "$STAGE1_CKPT" && test -s "$A1_CKPT" && test -s "$TRAIN_LIST" && test -s "$VAL_LIST"
    [[ "$(basename "$TRAIN_LIST")" != *test* && "$(basename "$VAL_LIST")" != *test* ]] || { echo "TEST forbidden"; exit 4; }
    echo "LATE DECISION PREFLIGHT PASSED"
    ;;
  cache)
    "$0" preflight
    FORCE_CACHE="${FORCE_CACHE:-0}"
    if [[ ! -s "$TRAIN_CACHE" || "$FORCE_CACHE" == "1" ]]; then
      py export_late_cache --checkpoint "$STAGE1_CKPT" --a1_ckpt "$A1_CKPT" --dataset_path "$DATASET_PATH" \
        --list_file "$TRAIN_LIST" --split train --output "$TRAIN_CACHE" --device "$DEVICE" --num_workers "${NUM_WORKERS:-4}"
    fi
    if [[ ! -s "$VAL_CACHE" || "$FORCE_CACHE" == "1" ]]; then
      py export_late_cache --checkpoint "$STAGE1_CKPT" --a1_ckpt "$A1_CKPT" --dataset_path "$DATASET_PATH" \
        --list_file "$VAL_LIST" --split val --output "$VAL_CACHE" --device "$DEVICE" --num_workers "${NUM_WORKERS:-4}"
    fi
    py check_identity --checkpoint "$STAGE1_CKPT" --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE"
    ;;
  identity)
    py check_identity --checkpoint "$STAGE1_CKPT" --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE"
    ;;
  oracle)
    py oracle_budget --val_cache "$VAL_CACHE" --output "$EVAL_ROOT/E0_oracle.json"
    ;;
  specialist)
    V="${ARG:-C3}"
    py train_specialist --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" \
      --output_dir "$SPEC_ROOT/${V}_seed${SEED}" --feature_variant "$V" --candidate base_intact \
      --steps "${SPEC_STEPS:-3000}" --eval_every "${SPEC_EVAL_EVERY:-250}" --device "$DEVICE" --seed "$SEED"
    ;;
  prototype)
    V="${ARG:-C3}"
    SCORE="$SPEC_ROOT/${V}_seed${SEED}/val_scores.pt"
    test -s "$SCORE" || { echo "Missing specialist score: $SCORE"; exit 5; }
    py prototype_verify --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" --specialist_scores "$SCORE" \
      --output "$EVAL_ROOT/E2_proto_${V}.json"
    ;;
  memory)
    CONTROL="${ARG:-none}"
    SCORE="$SPEC_ROOT/C3_seed${SEED}/val_scores.pt"
    test -s "$SCORE" || { echo "Missing C3 specialist score: $SCORE"; exit 5; }
    py memory_verify --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" --specialist_scores "$SCORE" \
      --output "$EVAL_ROOT/E3_memory_${CONTROL}.json" --control "$CONTROL" --device "$DEVICE"
    ;;
  status)
    find "$OUT_ROOT" "$EVAL_ROOT" -maxdepth 5 -type f \( -name '*.json' -o -name '*.pt' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort
    ;;
  *)
    echo "Usage: $0 {preflight|cache|identity|oracle|specialist C0|C1|C2|C3|prototype C3|memory none|shuffle_labels|shuffle_events|wrong_regions|status}" >&2
    exit 2
    ;;
esac
