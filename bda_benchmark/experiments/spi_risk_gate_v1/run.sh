#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-auto}"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-$BDA_ROOT/dataset}"
TRAIN_LIST="${TRAIN_LIST:-$DATASET_ROOT/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-$DATASET_ROOT/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
M4_CKPT="${M4_CKPT:-/root/data/BRIGHT-checkpoints/SPI_CORE_RING_ADAPTER_V1/seed3407/pilot500/joint_last.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-eval-results/SPI_RISK_GATE_V1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_RISK_GATE_V1/seed${SEED}}"

for path in "$DATASET_ROOT" "$TRAIN_LIST" "$VAL_LIST" "$P3_CKPT" "$M4_CKPT" "$SAR_STATS_JSON"; do
  test -e "$path" || { echo "Missing SPI Risk Gate input: $path" >&2; exit 1; }
done

if [[ "$MODE" == "preflight" ]]; then
  python -c "import cv2, scipy, torch; from bda_benchmark.experiments.spi_risk_gate_v1 import core, runtime; print('imports OK', torch.__version__)"
  python -m unittest discover -v "$BDA_ROOT/experiments/spi_risk_gate_v1/tests"
  bash -n "$BDA_ROOT/experiments/spi_risk_gate_v1/run.sh"
  echo "SPI Risk Gate V1 isolated preflight passed"
  exit 0
fi

python -u -m bda_benchmark.experiments.spi_risk_gate_v1.orchestrate \
  --mode "$MODE" \
  --base-checkpoint "$P3_CKPT" \
  --m4-checkpoint "$M4_CKPT" \
  --m4-steps "${M4_STEPS:-500}" \
  --dataset-path "$DATASET_ROOT" \
  --train-list "$TRAIN_LIST" \
  --val-list "$VAL_LIST" \
  --sar-stats-json "$SAR_STATS_JSON" \
  --output-root "$OUTPUT_ROOT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --device "${DEVICE:-cuda:0}" \
  --seed "$SEED" \
  --num-workers "${NUM_WORKERS:-4}" \
  --crop-size "${CROP_SIZE:-640}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
  --gate-steps "${GATE_STEPS:-500}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}"

