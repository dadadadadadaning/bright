#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit|relation|matching|slic) ;;
  *) echo "Choose preflight|smoke|audit|relation|matching|slic" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_evidence_v2"

if [[ "${MODE}" == "preflight" ]]; then
  python -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/build_cache.py" \
    "${SCRIPT_DIR}/audit_evidence.py" "${SCRIPT_DIR}/verifier.py" \
    "${SCRIPT_DIR}/train_verifier.py" "${SCRIPT_DIR}/audit_verifier.py" \
    "${SCRIPT_DIR}/select_admitted.py" "${SCRIPT_DIR}/summarize_autorun.py"
  python -c "import skimage; print('scikit-image', skimage.__version__)"
  python -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  bash -n "${SCRIPT_DIR}/watch_then_run.sh"
  echo "SPI S2 Evidence V2 preflight passed"
  exit 0
fi

DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
SEED="${SEED:-3407}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_UTILITY_GATE_V1/split_seed${SEED}}"
TRAIN_LIST="${TRAIN_LIST:-${SPLIT_ROOT}/gate_train.txt}"
CALIBRATION_LIST="${CALIBRATION_LIST:-${SPLIT_ROOT}/gate_calibration.txt}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_EVIDENCE_V2/cache_seed${SEED}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_S2_EVIDENCE_V2}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_EVIDENCE_V2}"
TRAIN_CACHE="${CACHE_ROOT}/train.pt"
CALIBRATION_CACHE="${CACHE_ROOT}/internal_calibration.pt"
ADMISSION_JSON="${RESULT_ROOT}/parameter_free_admission_seed${SEED}.json"
DEFAULT_EVIDENCE_GROUP_LIST="direct,relation,matching,grid,slic"
# Some older experiment launchers export EVIDENCE_GROUPS=0 as a Boolean switch.
# In this program the variable contains a comma-separated group list, so treat
# legacy Boolean values as "use the complete default audit" instead of passing
# a fake group named 0/1 to build_cache.py.
case "${EVIDENCE_GROUPS:-}" in
  ""|0|1|false|true|False|True) EVIDENCE_GROUP_LIST="${DEFAULT_EVIDENCE_GROUP_LIST}" ;;
  *) EVIDENCE_GROUP_LIST="${EVIDENCE_GROUPS}" ;;
esac

for required in "${DATASET_ROOT}" "${P3_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${CALIBRATION_LIST}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
mkdir -p "${CACHE_ROOT}" "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"

if [[ "${MODE}" == "smoke" ]]; then
  python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.build_cache \
    --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${TRAIN_LIST}" --sar-stats-json "${SAR_STATS_JSON}" \
    --output "${CACHE_ROOT}/smoke.pt" --split train --groups "${EVIDENCE_GROUP_LIST}" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-2}" \
    --expected-images "${EXPECTED_TRAIN_IMAGES:-2627}" \
    --max-images "${SMOKE_IMAGES:-10}" --maximum-per-class-per-image 128 \
    --matching-max-side "${MATCHING_MAX_SIDE:-48}" \
    --matching-window "${MATCHING_WINDOW:-5}" --cell-size "${CELL_SIZE:-4}" \
    --seed "${SEED}"
  echo "SPI S2 Evidence V2 real-data smoke passed. Val450 was not accessed."
  exit 0
fi

build_caches_and_audit() {
  if [[ ! -f "${TRAIN_CACHE}" ]]; then
    python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.build_cache \
      --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
      --list-path "${TRAIN_LIST}" --sar-stats-json "${SAR_STATS_JSON}" \
      --output "${TRAIN_CACHE}" --split train --groups "${EVIDENCE_GROUP_LIST}" \
      --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-4}" \
      --expected-images "${EXPECTED_TRAIN_IMAGES:-2627}" \
      --maximum-per-class-per-image "${MAX_PER_CLASS_PER_IMAGE:-2048}" \
      --matching-max-side "${MATCHING_MAX_SIDE:-80}" \
      --matching-window "${MATCHING_WINDOW:-5}" --cell-size "${CELL_SIZE:-4}" \
      --seed "${SEED}"
  fi
  if [[ ! -f "${CALIBRATION_CACHE}" ]]; then
    python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.build_cache \
      --checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" \
      --list-path "${CALIBRATION_LIST}" --sar-stats-json "${SAR_STATS_JSON}" \
      --output "${CALIBRATION_CACHE}" --split internal_calibration --groups "${EVIDENCE_GROUP_LIST}" \
      --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-4}" \
      --expected-images "${EXPECTED_CALIBRATION_IMAGES:-291}" \
      --maximum-per-class-per-image 0 --matching-max-side "${MATCHING_MAX_SIDE:-80}" \
      --matching-window "${MATCHING_WINDOW:-5}" --cell-size "${CELL_SIZE:-4}" \
      --seed "${SEED}"
  fi
  python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CALIBRATION_CACHE}" \
    --output "${ADMISSION_JSON}"
}

if [[ "${MODE}" == "audit" ]]; then
  build_caches_and_audit
  echo "Parameter-free evidence audit complete. Val450 was not accessed."
  exit 0
fi

test -f "${TRAIN_CACHE}" || { echo "Run audit mode first: missing ${TRAIN_CACHE}" >&2; exit 1; }
test -f "${CALIBRATION_CACHE}" || { echo "Run audit mode first: missing ${CALIBRATION_CACHE}" >&2; exit 1; }
test -f "${ADMISSION_JSON}" || { echo "Run audit mode first: missing ${ADMISSION_JSON}" >&2; exit 1; }

VERIFIER_DIR="${CHECKPOINT_ROOT}/${MODE}_verifier_step${VERIFIER_STEPS:-500}_seed${SEED}"
python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.train_verifier \
  --group "${MODE}" --train-cache "${TRAIN_CACHE}" \
  --admission-json "${ADMISSION_JSON}" --output-dir "${VERIFIER_DIR}" \
  --device "${DEVICE:-cuda:0}" --seed "${SEED}" \
  --optimizer-steps "${VERIFIER_STEPS:-500}" \
  --batch-size "${VERIFIER_BATCH_SIZE:-8192}" \
  --learning-rate "${VERIFIER_LR:-1e-3}"

python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.audit_verifier \
  --calibration-cache "${CALIBRATION_CACHE}" \
  --verifier-checkpoint "${VERIFIER_DIR}/verifier_last.pth" \
  --output "${RESULT_ROOT}/${MODE}_verifier_internal_step${VERIFIER_STEPS:-500}_seed${SEED}.json" \
  --device "${DEVICE:-cuda:0}" --expected-steps "${VERIFIER_STEPS:-500}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"

echo "Single-evidence ${MODE} verifier complete. Val450 was not accessed."
