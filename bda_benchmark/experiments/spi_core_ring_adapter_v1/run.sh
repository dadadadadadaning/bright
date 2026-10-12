#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-auto}"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
TRAIN_LIST="${TRAIN_LIST:-$DATASET_ROOT/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-$DATASET_ROOT/splitname/standard_ML/val_set.txt}"
BASE_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SEED="${SEED:-3407}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CORE_RING_ADAPTER_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CORE_RING_ADAPTER_V1}"

required=("$DATASET_ROOT" "$TRAIN_LIST" "$VAL_LIST" "$BASE_CKPT" "$SAR_STATS_JSON")
for path in "${required[@]}"; do
  test -e "$path" || { echo "Missing required input: $path" >&2; exit 1; }
done

if [[ "$MODE" == "preflight" ]]; then
  python -c "import cv2, scipy, torch; from bda_benchmark.experiments.spi_core_ring_adapter_v1 import model, regions; print('imports OK', torch.__version__)"
  python -m unittest discover -v "$BDA_ROOT/experiments/spi_core_ring_adapter_v1/tests"
  bash -n "$BDA_ROOT/experiments/spi_core_ring_adapter_v1/run.sh"
  echo "SPI Core/Ring adapter isolated preflight passed"
  exit 0
fi

python -u -m bda_benchmark.experiments.spi_core_ring_adapter_v1.orchestrate \
  --mode "$MODE" \
  --dataset-path "$DATASET_ROOT" \
  --train-list "$TRAIN_LIST" \
  --val-list "$VAL_LIST" \
  --base-checkpoint "$BASE_CKPT" \
  --sar-stats-json "$SAR_STATS_JSON" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --result-root "$RESULT_ROOT" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:0}" \
  --crop-size "${CROP_SIZE:-640}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --pilot-steps "${PILOT_STEPS:-500}" \
  --full-steps "${FULL_STEPS:-3000}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}"

