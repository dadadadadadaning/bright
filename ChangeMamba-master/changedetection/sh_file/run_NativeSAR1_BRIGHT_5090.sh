#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master/ChangeMamba-master}"
CHANGEDIR="${PROJECT_ROOT}/changedetection"
DATA_ROOT="${DATA_ROOT:-/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset}"
LIST_ROOT="${LIST_ROOT:-${CHANGEDIR}/datasets/data_name_list/BRIGHT}"
CFG_PATH="${CFG_PATH:-${CHANGEDIR}/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
SAR_STATS="${SAR_STATS:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
BASELINE_CKPT="${BASELINE_CKPT:-${PROJECT_ROOT}/results/checkpoints_B0_full/BRIGHT/ChangeMamba-MMBDA_B0_cat_1783259203.4570115/best_model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/native_sar1}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/NativeSAR1_inputonly3000_seed3407}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${OUTPUT_ROOT}/NativeSAR1_smoke20_seed3407}"
SEED="${SEED:-3407}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${CHANGEDIR}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    changedetection/experiments/native_sar/model.py \
    changedetection/experiments/native_sar/dataset.py \
    changedetection/experiments/native_sar/trainer.py \
    changedetection/script/train_ChangeMamba_native_sar.py \
    changedetection/script/eval_ChangeMamba_native_sar.py
  python -m unittest changedetection.tests.test_native_sar_experiment -v
  echo "ChangeMamba NativeSAR1 preflight passed."
}

train_run() {
  local steps="$1"
  local eval_every="$2"
  local run_output="$3"
  test -s "${BASELINE_CKPT}"
  test -s "${SAR_STATS}"
  mkdir -p "${run_output}"
  python -u changedetection/script/train_ChangeMamba_native_sar.py \
    --cfg "${CFG_PATH}" \
    --baseline_checkpoint "${BASELINE_CKPT}" \
    --sar_stats_path "${SAR_STATS}" \
    --output_dir "${run_output}" \
    --train_dataset_path "${DATA_ROOT}" \
    --train_data_list_path "${LIST_ROOT}/train_set.txt" \
    --val_dataset_path "${DATA_ROOT}" \
    --val_data_list_path "${LIST_ROOT}/val_set.txt" \
    --train_batch_size 2 \
    --eval_batch_size 4 \
    --num_workers 4 \
    --crop_size 512 \
    --max_iters "${steps}" \
    --eval_every "${eval_every}" \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --expected_baseline_miou 64.2585 \
    --baseline_tolerance_pp 0.05 \
    --seed "${SEED}"
}

eval_run() {
  local split="$1"
  local list_path="${LIST_ROOT}/val_set.txt"
  local extra=()
  local ckpt="${NATIVE_SAR_CKPT:-${TRAIN_OUTPUT_DIR}/best_model.pth}"
  if [[ "${split}" == "test" ]]; then
    if [[ "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Test locked. Set ALLOW_TEST=1 only after best_model.pth is frozen." >&2
      exit 1
    fi
    list_path="${LIST_ROOT}/test_set.txt"
    extra+=(--allow_test)
  fi
  test -s "${ckpt}"
  python -u changedetection/script/eval_ChangeMamba_native_sar.py \
    --cfg "${CFG_PATH}" \
    --checkpoint "${ckpt}" \
    --sar_stats_path "${SAR_STATS}" \
    --dataset_path "${DATA_ROOT}" \
    --data_list_path "${list_path}" \
    --split "${split}" \
    --output_json "${TRAIN_OUTPUT_DIR}/independent_${split}.json" \
    --crop_size 512 \
    --batch_size 4 \
    --num_workers 2 \
    "${extra[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) train_run 20 20 "${SMOKE_OUTPUT_DIR}" ;;
  train3000) train_run 3000 500 "${TRAIN_OUTPUT_DIR}" ;;
  eval_val) eval_run val ;;
  eval_test) eval_run test ;;
  *) echo "Usage: $0 preflight|smoke|train3000|eval_val|eval_test" >&2; exit 2 ;;
esac
