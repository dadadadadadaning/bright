#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
# Default to the validation-qualified practical-mask readout that the current
# experiment chain has already produced.  Oracle/A1 checkpoints can still be
# supplied explicitly through Q0_CKPT for controlled comparisons.
Q0_CKPT="${Q0_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/Q0_SAR_READOUT/Q0_B0_screen1500_seed3407/best_readout.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/Q1_REGION_EVIDENCE}"
SUPPORT="${SUPPORT:-p3}"
SEED="${SEED:-3407}"

case "${MODE}" in
  preflight)
    bash -n run/run_Q1_region_evidence.sh
    python -m py_compile \
      script/standard_ML/q1_r0_runtime.py \
      script/standard_ML/audit_Q1_R0_region_aggregation.py
    python -m unittest -v tests.test_q1_r0_region_aggregation
    echo "Q1-R0 region-evidence preflight passed."
    exit 0
    ;;
  r0_smoke)
    MAX_IMAGES="${MAX_IMAGES:-10}"
    TAG="smoke"
    ;;
  r0_val)
    MAX_IMAGES=0
    TAG="val"
    ;;
  eval_test)
    echo "Q1-R0 is validation-only; test access is forbidden." >&2
    exit 2
    ;;
  *) echo "Usage: $0 preflight|r0_smoke|r0_val" >&2; exit 2;;
esac

for required in "${VAL_LIST}" "${SAR_STATS_JSON}" "${H6_A2_CKPT}" "${Q0_CKPT}"; do
  test -s "${required}" || { echo "Missing Q1-R0 input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/Q1_R0_${SUPPORT}_${TAG}_seed${SEED}.json}"

echo "================================================================================"
echo "Protocol         : Q1-R0 parameter-free region aggregation"
echo "Q0 readout       : ${Q0_CKPT}"
echo "Region support   : ${SUPPORT}"
echo "Methods          : pixel_vote/mean/max/top10/top20/lse/area"
echo "Training         : NONE"
echo "Test accessed    : NEVER"
echo "Output           : ${OUTPUT_JSON}"
echo "================================================================================"

python -u script/standard_ML/audit_Q1_R0_region_aggregation.py \
  --q0_checkpoint "${Q0_CKPT}" --source_checkpoint "${H6_A2_CKPT}" \
  --dataset_path "${DATASET_ROOT}" --data_list "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --output_json "${OUTPUT_JSON}" \
  --support "${SUPPORT}" --p3_threshold "${P3_THRESHOLD:-0.5}" \
  --minimum_pixels "${MINIMUM_PIXELS:-16}" \
  --minimum_building_fraction "${MINIMUM_BUILDING_FRACTION:-0.5}" \
  --minimum_class_purity "${MINIMUM_CLASS_PURITY:-0.75}" \
  --lse_beta "${LSE_BETA:-10.0}" --area_threshold "${AREA_THRESHOLD:-0.5}" \
  --device cuda:0 --crop_size 640 --num_workers "${NUM_WORKERS:-8}" \
  --max_images "${MAX_IMAGES}" --seed "${SEED}"
