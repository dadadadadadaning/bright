#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-auto}"
case "${MODE}" in
  preflight|smoke|auto|fit-policy|train-only) ;;
  *) echo "Choose preflight|smoke|auto|fit-policy|train-only" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_utility_gate_v1"

if [[ "${MODE}" == "preflight" ]]; then
  python -m py_compile \
    "${SCRIPT_DIR}/core.py" \
    "${SCRIPT_DIR}/runtime.py" \
    "${SCRIPT_DIR}/fit_direction_policy.py" \
    "${SCRIPT_DIR}/train_gate.py" \
    "${SCRIPT_DIR}/audit.py" \
    "${SCRIPT_DIR}/orchestrate.py"
  python -m unittest discover -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "SPI S2 Utility Gate V1 preflight passed"
  exit 0
fi

DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_UTILITY_GATE_V1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1}"

for required in "${DATASET_ROOT}" "${TRAIN_LIST}" "${VAL_LIST}" "${P3_CKPT}" "${SAR_STATS_JSON}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done

python -u -m bda_benchmark.experiments.spi_s2_utility_gate_v1.orchestrate \
  --mode "${MODE}" \
  --checkpoint "${P3_CKPT}" \
  --dataset-path "${DATASET_ROOT}" \
  --train-list "${TRAIN_LIST}" \
  --val-list "${VAL_LIST}" \
  --sar-stats-json "${SAR_STATS_JSON}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --device "${DEVICE:-cuda:0}" \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --optimizer-steps "${PILOT_STEPS:-500}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --minimum-area "${MINIMUM_AREA:-32}" \
  --gate-threshold "${GATE_THRESHOLD:-0.50}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
  --expected-p3-miou "${EXPECTED_P3_MIOU:-0.605020}" \
  --policy-max-images "${POLICY_MAX_IMAGES:--1}"
