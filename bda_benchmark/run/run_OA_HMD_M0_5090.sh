#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/OA_HMD}"
P3_BASELINE_JSON="${P3_BASELINE_JSON:-/root/shared-nvme/BRIGHT-checkpoints/OA_HMD/OA_HMD_C0_guard_seed3407/independent_val.json}"
BETAS="${BETAS:--0.50 -0.40 -0.30 -0.20 -0.10 0.00 0.10 0.20 0.30 0.40 0.50}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_OA_HMD_M0_5090.sh
    python -m py_compile \
      script/standard_ML/m0_boundary_runtime.py \
      script/standard_ML/audit_OA_HMD_M0_boundary.py
    python -m unittest -v tests.test_oa_hmd_m0_boundary
    echo "OA-HMD M0 preflight passed."
    exit 0
    ;;
  smoke)
    MAX_BATCHES="${MAX_BATCHES:-2}"
    OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/OA_HMD_M0_boundary_smoke_seed${SEED}.json}"
    ;;
  val)
    MAX_BATCHES=0
    OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/OA_HMD_M0_boundary_val_seed${SEED}.json}"
    ;;
  *)
    echo "Usage: $0 {preflight|smoke|val}" >&2
    exit 2
    ;;
esac

for required in "${P3_CKPT}" "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing OA-HMD M0 input: ${required}" >&2
    exit 1
  }
done
mkdir -p "$(dirname "${OUTPUT_JSON}")"
BASELINE_ARGS=()
if [[ "${MODE}" == "val" && -s "${P3_BASELINE_JSON}" ]]; then
  BASELINE_ARGS=(--expected_baseline_json "${P3_BASELINE_JSON}")
fi

echo "================================================================================"
echo "Protocol       : OA-HMD M0 validation-only boundary audit"
echo "P3 checkpoint  : ${P3_CKPT}"
echo "Support        : frozen hard P3 building prediction"
echo "Offsets        : ${BETAS}"
if [[ "${MODE}" == "val" ]]; then
  echo "Exact guard    : ${P3_BASELINE_JSON}"
else
  echo "Exact guard    : skipped for partial smoke"
fi
echo "Training       : NONE"
echo "Test accessed  : NEVER"
echo "Output         : ${OUTPUT_JSON}"
echo "================================================================================"

python -u script/standard_ML/audit_OA_HMD_M0_boundary.py \
  --checkpoint "${P3_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --data_list_path "${VAL_LIST}" \
  --output_json "${OUTPUT_JSON}" \
  --device cuda:0 --crop_size 640 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --max_batches "${MAX_BATCHES}" \
  --betas "${BETAS}" --seed "${SEED}" \
  "${BASELINE_ARGS[@]}"
