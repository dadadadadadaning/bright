#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
A2_CKPT="${A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
Q0_CKPT="${Q0_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/Q0_SAR_READOUT/Q0_B0_screen1500_seed3407/best_readout.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/Q2_BUILDING_RESIDUAL}"
SEED="${SEED:-3407}"

common=(
  --p3_checkpoint "${P3_CKPT}"
  --a2_checkpoint "${A2_CKPT}"
  --q0_checkpoint "${Q0_CKPT}"
  --dataset_path "${DATASET_ROOT}"
  --train_list "${TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --sar_stats_path "${SAR_STATS_JSON}"
  --device cuda:0
  --crop_size "${CROP_SIZE:-640}"
  --eval_batch_size 1
  --num_workers "${NUM_WORKERS:-8}"
  --proposal_threshold "${PROPOSAL_THRESHOLD:-0.5}"
  --minimum_area "${MINIMUM_AREA:-16}"
  --minimum_building_fraction "${MINIMUM_BUILDING_FRACTION:-0.5}"
  --minimum_class_purity "${MINIMUM_CLASS_PURITY:-0.75}"
  --maximum_transition "${MAXIMUM_TRANSITION:-0.25}"
  --seed "${SEED}"
)

case "${MODE}" in
  preflight)
    bash -n run/run_Q2_building_residual.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_Q2_BuildingResidual.py \
      script/standard_ML/q2_building_residual_runtime.py \
      script/standard_ML/train_Q2_building_residual.py \
      script/standard_ML/eval_Q2_building_residual.py
    python -m unittest -v tests.test_q2_building_residual
    echo "Q2 building-residual preflight passed."
    exit 0
    ;;
  guard)
    VARIANT=q2_b0
    EXP_NAME="${EXP_NAME:-Q2_B0_guard_seed${SEED}}"
    STEPS=0
    EVAL_EVERY=1
    VAL_MAX_BATCHES=0
    ;;
  smoke)
    VARIANT=q2_b1
    EXP_NAME="${EXP_NAME:-Q2_B1_smoke20_seed${SEED}}"
    STEPS="${OPTIMIZER_STEPS:-20}"
    EVAL_EVERY="${EVAL_EVERY:-20}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    ;;
  screen)
    VARIANT=q2_b1
    EXP_NAME="${EXP_NAME:-Q2_B1_screen1500_seed${SEED}}"
    STEPS="${OPTIMIZER_STEPS:-1500}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    VAL_MAX_BATCHES=0
    ;;
  eval_val)
    Q2_CKPT="${Q2_CKPT:-}"
    test -s "${Q2_CKPT}" || { echo "Missing Q2_CKPT: ${Q2_CKPT}" >&2; exit 1; }
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-$(dirname "${Q2_CKPT}")/independent_val.json}"
    python -u script/standard_ML/eval_Q2_building_residual.py \
      --checkpoint "${Q2_CKPT}" \
      --p3_checkpoint "${P3_CKPT}" --a2_checkpoint "${A2_CKPT}" \
      --q0_checkpoint "${Q0_CKPT}" --dataset_path "${DATASET_ROOT}" \
      --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size "${CROP_SIZE:-640}" \
      --eval_batch_size 1 --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  eval_test)
    echo "Q2 test access is forbidden until B1 passes the validation gate." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 preflight|guard|smoke|screen|eval_val" >&2
    exit 2
    ;;
esac

for required in "${TRAIN_LIST}" "${VAL_LIST}" "${SAR_STATS_JSON}" \
  "${P3_CKPT}" "${A2_CKPT}" "${Q0_CKPT}"; do
  test -s "${required}" || { echo "Missing Q2 input: ${required}" >&2; exit 1; }
done

OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args+=(--resume "${RESUME}")
fi
echo "================================================================================"
echo "Protocol         : Q2 exact-P3 building-region residual"
echo "Variant          : ${VARIANT}"
echo "P3/A2/Q0         : frozen / frozen / frozen"
echo "Region source    : frozen P3 connected components"
echo "Trainable        : 18D region MLP + zero transition only"
echo "Selector         : event-worst -> event-mean -> Damaged IoU -> valid mIoU"
echo "Training/test    : ${STEPS} steps / NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "================================================================================"

python -u script/standard_ML/train_Q2_building_residual.py \
  --variant "${VARIANT}" "${common[@]}" --output_dir "${OUTPUT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-8}" \
  --optimizer_steps "${STEPS}" --eval_every "${EVAL_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --learning_rate "${LEARNING_RATE:-0.001}" \
  --weight_decay "${WEIGHT_DECAY:-0.0001}" \
  --focus_probability "${FOCUS_PROBABILITY:-0.5}" \
  --val_max_batches "${VAL_MAX_BATCHES}" "${resume_args[@]}"
