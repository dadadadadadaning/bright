#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-smoke}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_CKPT="${H6_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_JSON="${OUTPUT_JSON:-/root/shared-nvme/BRIGHT-eval-results/H6_SDCR/H6_A2_information_funnel_${MODE}.json}"

case "${MODE}" in
  preflight)
    python -m py_compile script/standard_ML/audit_H6_information_funnel.py
    python -m unittest -v tests.test_h6_information_funnel
    exit 0
    ;;
  smoke)
    LIMIT_ARGS=(--max_images "${MAX_IMAGES:-8}" --num_workers "${NUM_WORKERS:-2}")
    ;;
  val)
    LIMIT_ARGS=(--num_workers "${NUM_WORKERS:-8}")
    ;;
  *) echo "Expected preflight|smoke|val" >&2; exit 2 ;;
esac

for required in "${VAL_LIST}" "${SAR_STATS_JSON}" "${H6_CKPT}"; do
  test -s "${required}" || { echo "Missing information-funnel input: ${required}" >&2; exit 1; }
done

python -u script/standard_ML/audit_H6_information_funnel.py \
  --checkpoint "${H6_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_list "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${EVAL_BATCH_SIZE:-1}" \
  "${LIMIT_ARGS[@]}"
