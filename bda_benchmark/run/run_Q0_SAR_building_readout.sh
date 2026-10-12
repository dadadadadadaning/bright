#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

VARIANT="${VARIANT:-q0_a0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/Q0_SAR_READOUT}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${MODE}" in
  preflight)
    bash -n run/run_Q0_SAR_building_readout.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_Q0_Readout.py \
      script/standard_ML/q0_readout_runtime.py \
      script/standard_ML/train_Q0_SAR_building_readout.py \
      script/standard_ML/eval_Q0_SAR_building_readout.py
    python -m unittest -v tests.test_q0_sar_building_readout
    echo "Q0 SAR building-readout preflight passed."
    exit 0
    ;;
  smoke)
    STEPS="${OPTIMIZER_STEPS:-20}"
    EVAL_EVERY="${EVAL_EVERY:-20}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    COUNT_MAX_BATCHES="${COUNT_MAX_BATCHES:-4}"
    TAG="smoke${STEPS}"
    ;;
  screen|train)
    STEPS="${OPTIMIZER_STEPS:-1500}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    VAL_MAX_BATCHES=0
    COUNT_MAX_BATCHES=0
    TAG="screen${STEPS}"
    ;;
  eval_val)
    Q0_CKPT="${Q0_CKPT:-}"
    test -n "${Q0_CKPT}" || { echo "eval_val requires Q0_CKPT" >&2; exit 1; }
    for required in "${Q0_CKPT}" "${H6_A2_CKPT}" "${VAL_LIST}" "${SAR_STATS_JSON}"; do
      test -s "${required}" || { echo "Missing Q0 validation input: ${required}" >&2; exit 1; }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${Q0_CKPT%/*}/independent_val.json}"
    python -u script/standard_ML/eval_Q0_SAR_building_readout.py \
      --checkpoint "${Q0_CKPT}" --source_checkpoint "${H6_A2_CKPT}" \
      --dataset_path "${DATASET_ROOT}" --data_list "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --split val --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-8}" \
      --seed "${SEED}"
    exit 0
    ;;
  eval_test)
    echo "Q0 is validation-only by contract; eval_test is intentionally forbidden." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 preflight|smoke|screen|train|eval_val" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in q0_a0|q0_a1|q0_b0) ;; *) echo "Invalid Q0 variant: ${VARIANT}" >&2; exit 2;; esac
for required in "${H6_A2_CKPT}" "${TRAIN_LIST}" "${VAL_LIST}" "${SAR_STATS_JSON}"; do
  test -s "${required}" || { echo "Missing Q0 input: ${required}" >&2; exit 1; }
done
if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed Q0 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Locked source    : ${H6_A2_CKPT}"
echo "Trainable module : one Stage4 C->3 readout only"
echo "Building mask    : $([[ ${VARIANT} == q0_b0 ]] && echo frozen-P3 || echo oracle-GT)"
echo "Class balance    : $([[ ${VARIANT} == q0_a1 ]] && echo effective-number || echo none)"
echo "Final H6 output  : NEVER modified"
echo "Test access      : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "================================================================================"

python -u script/standard_ML/train_Q0_SAR_building_readout.py \
  --variant "${VARIANT}" --source_checkpoint "${H6_A2_CKPT}" \
  --dataset_path "${DATASET_ROOT}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 --crop_size 640 --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${STEPS}" --eval_every "${EVAL_EVERY}" --log_every "${LOG_EVERY:-20}" \
  --num_workers "${NUM_WORKERS:-8}" --learning_rate "${NEW_LR:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --focus_probability "${FOCUS_PROBABILITY:-0.5}" \
  --mask_threshold "${P3_MASK_THRESHOLD:-0.5}" --minimum_purity "${MINIMUM_PURITY:-0.75}" \
  --class_balance_beta "${CLASS_BALANCE_BETA:-0.9999}" \
  --maximum_weight_ratio "${MAXIMUM_WEIGHT_RATIO:-5.0}" \
  --count_max_batches "${COUNT_MAX_BATCHES}" --val_max_batches "${VAL_MAX_BATCHES}" \
  --seed "${SEED}" "${RESUME_ARGS[@]}"
