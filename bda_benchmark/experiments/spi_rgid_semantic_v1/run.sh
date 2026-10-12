#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|controls|train-online|rgid) ;;
  *) echo "Choose preflight|controls|train-online|rgid" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
CONTRACT="${SCRIPT_DIR}/experiment_contract.json"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

verify=(python -m bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.verify_contract
  --contract "${CONTRACT}" --project-root "${PROJECT_ROOT}")

if [[ "${MODE}" == "preflight" ]]; then
  "${verify[@]}"
  bash -n "${SCRIPT_DIR}/run.sh"
  bash -n run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh
  python -m py_compile \
    "${SCRIPT_DIR}/method.py" \
    "${SCRIPT_DIR}/heads.py" \
    "${SCRIPT_DIR}/fast_targets.py" \
    "${SCRIPT_DIR}/training.py" \
    "${SCRIPT_DIR}/objectives.py" \
    "${SCRIPT_DIR}/event_protocol.py" \
    "${SCRIPT_DIR}/scripts/build_event_folds.py" \
    "${SCRIPT_DIR}/scripts/summarize_c3.py"
  python -m py_compile "${SCRIPT_DIR}/scripts/train_online_rgid.py"
  python -m unittest discover \
    -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
  echo "SPI-RGID semantic V1 isolated preflight passed"
  exit 0
fi

if [[ "${MODE}" == "train-online" ]]; then
  "${verify[@]}"
  STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
  DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
  TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
  P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
  RGID_OUTPUT_DIR="${RGID_OUTPUT_DIR:-${STORAGE_ROOT}/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407}"
  python -u -m bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.train_online_rgid \
    --checkpoint "${P3_CKPT}" \
    --dataset-path "${DATASET_ROOT}" \
    --train-list "${TRAIN_LIST}" \
    --output-dir "${RGID_OUTPUT_DIR}" \
    --device cuda:0 \
    --steps "${RGID_STEPS:-500}" \
    --batch-size "${RGID_BATCH_SIZE:-1}" \
    --accumulation "${RGID_ACCUMULATION:-4}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --calibration-samples "${CALIBRATION_SAMPLES:-256}" \
    --seed 3407
  exit 0
fi

if [[ "${MODE}" == "controls" ]]; then
  "${verify[@]}"
  env -u RGID_CKPT -u RGID_PROVENANCE_JSON \
    bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh audit
  exit 0
fi

: "${RGID_CKPT:?RGID_CKPT is required for rgid mode}"
: "${RGID_PROVENANCE_JSON:?RGID_PROVENANCE_JSON is required for rgid mode}"
test -s "${RGID_CKPT}" || { echo "Missing RGID checkpoint: ${RGID_CKPT}" >&2; exit 1; }
test -s "${RGID_PROVENANCE_JSON}" || { echo "Missing provenance: ${RGID_PROVENANCE_JSON}" >&2; exit 1; }
"${verify[@]}" --provenance "${RGID_PROVENANCE_JSON}"
bash run/run_SPI_GEOMETRY_CAUSAL_VAL450.sh audit
