#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|pilot|fsa|rcm) ;;
  *) echo "Choose preflight|pilot|fsa|rcm" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_sar_adapter_v1"

if [[ "${MODE}" == "preflight" ]]; then
  python -m py_compile "${SCRIPT_DIR}/model.py" "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/audit.py"
  python -m pytest -q "${SCRIPT_DIR}/test_identity.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "SPI SAR Adapter V1 preflight passed"
  exit 0
fi

DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1/split_seed3407}"
TRAIN_LIST="${TRAIN_LIST:-${SPLIT_ROOT}/gate_train.txt}"
CALIBRATION_LIST="${CALIBRATION_LIST:-${SPLIT_ROOT}/gate_calibration.txt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_SAR_ADAPTER_V1}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_SAR_ADAPTER_V1}"
STEPS="${PILOT_STEPS:-500}"
SEED="${SEED:-3407}"

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${CALIBRATION_LIST}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"

run_variant() {
  local variant="$1"
  local output_dir="${CHECKPOINT_ROOT}/${variant}_step${STEPS}_seed${SEED}"
  python -u -m bda_benchmark.experiments.spi_sar_adapter_v1.train \
    --variant "${variant}" --base-checkpoint "${P3_CKPT}" \
    --dataset-path "${DATASET_ROOT}" --train-list "${TRAIN_LIST}" \
    --output-dir "${output_dir}" --device "${DEVICE:-cuda:0}" \
    --seed "${SEED}" --batch-size "${BATCH_SIZE:-4}" \
    --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
    --optimizer-steps "${STEPS}" --num-workers "${NUM_WORKERS:-4}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --lambda-anchor "${LAMBDA_ANCHOR:-0.10}"

  python -u -m bda_benchmark.experiments.spi_sar_adapter_v1.audit \
    --variant "${variant}" --base-checkpoint "${P3_CKPT}" \
    --adapter-checkpoint "${output_dir}/adapter_last.pth" \
    --dataset-path "${DATASET_ROOT}" --list-path "${CALIBRATION_LIST}" \
    --sar-stats-json "${SAR_STATS_JSON}" \
    --output-json "${RESULT_ROOT}/${variant}_internal_step${STEPS}_seed${SEED}.json" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-4}" \
    --expected-images "${EXPECTED_CALIBRATION_IMAGES:-291}" \
    --expected-steps "${STEPS}" --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed "${SEED}"
}

if [[ "${MODE}" == "pilot" || "${MODE}" == "fsa" ]]; then run_variant fsa; fi
if [[ "${MODE}" == "pilot" || "${MODE}" == "rcm" ]]; then run_variant rcm; fi

echo "SAR adapter run complete. Val450 was not accessed."
