#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
BASE_CKPT="${BASE_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/data/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/S8_P5_HIERARCHICAL}"
SEED="${SEED:-7341}"
VARIANT="${VARIANT:-c2}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile model/DamageFormer_S8_P5_Hierarchical.py \
    script/standard_ML/train_S8_P5_hierarchical.py \
    script/standard_ML/eval_S8_P5_hierarchical.py
  python -m unittest tests/test_s8_p5_hierarchical.py -v
  echo "S8-P5-H preflight passed"
}

frozen_eval() {
  local split="$1"
  local checkpoint="${CHECKPOINT:-}"
  test -s "${checkpoint}" || {
    echo "Set CHECKPOINT to a validation-selected best_val_miou.pth" >&2
    exit 5
  }
  local list="${VAL_LIST}"
  if [[ "${split}" == "test" ]]; then list="${TEST_LIST}"; fi
  local output_json="${checkpoint%/*}/${split}_frozen_metrics.json"
  python -u script/standard_ML/eval_S8_P5_hierarchical.py \
    --checkpoint "${checkpoint}" --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${list}" --sar_stats_json "${SAR_STATS_JSON}" \
    --split "${split}" --output_json "${output_json}" \
    --batch_size 1 --num_workers 4 --device cuda:0
}

train_run() {
  local steps="$1" eval_every="$2" warmup="$3" tag="$4"
  local output="${OUTPUT_ROOT}/${VARIANT}_${tag}_seed${SEED}"
  mkdir -p "${output}"
  python -u script/standard_ML/train_S8_P5_hierarchical.py \
    --variant "${VARIANT}" --recovery_variant none \
    --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" --base_ckpt "${BASE_CKPT}" \
    --output_dir "${output}" --crop_size 640 --train_batch_size 2 \
    --eval_batch_size 1 --accumulation_steps 8 --optimizer_steps "${steps}" \
    --warmup_steps "${warmup}" --num_workers 4 --device cuda:0 --amp_dtype bf16 \
    --seed "${SEED}" --sar_learning_rate 1e-4 --fusion_learning_rate 1e-4 \
    --core_learning_rate 1e-6 --eval_every "${eval_every}" --save_every 500 \
    --guard_start_step 1500 --early_stop_drop_pp 1.0 --early_stop_patience 2
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) train_run 20 10 5 smoke20 ;;
  pilot) train_run 3000 250 500 pilot3000 ;;
  full) train_run 8000 500 1000 full8000 ;;
  val) frozen_eval val ;;
  test) frozen_eval test ;;
  *) echo "choose preflight|smoke|pilot|full|val|test" >&2; exit 2 ;;
esac

# TRAIN_LIST is always the original full training list.  VAL_LIST selects the
# best checkpoint.  TEST_LIST is intentionally not consumed during training;
# export test predictions only after the validation-selected model is frozen.
