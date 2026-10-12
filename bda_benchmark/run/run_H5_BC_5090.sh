#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT_WAS_SET="${VARIANT+x}"
VARIANT="${VARIANT:-b1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BC}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H5_BC}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
SOURCE_A4_CKPT="${SOURCE_A4_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth}"
JOINT_PRETRAIN_DIR="${JOINT_PRETRAIN_DIR:-${OUTPUT_ROOT}/H5_C1_joint_pretrain_seed${SEED}}"
JOINT_PRETRAIN_CKPT="${JOINT_PRETRAIN_CKPT:-${JOINT_PRETRAIN_DIR}/joint_adapters_final.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H5_BC_5090.sh
    bash -n run/run_H5_BC_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H5_BC.py \
      script/standard_ML/pretrain_H5_C_joint_rgbsar.py \
      script/standard_ML/train_DamageFormer_RGBSAR_H5_BC.py \
      script/standard_ML/summarize_H5_BC_results.py
    python -m unittest -v tests.test_damageformer_rgbsar_h5_bc
    echo "H5-B/C preflight passed."
    exit 0
    ;;
  summary)
    mkdir -p "${OUTPUT_ROOT}"
    python -u script/standard_ML/summarize_H5_BC_results.py \
      --output_root "${OUTPUT_ROOT}" \
      --source_a4_checkpoint "${SOURCE_A4_CKPT}" \
      --output_json "${OUTPUT_ROOT}/H5_BC_validation_comparison.json"
    exit 0
    ;;
  joint_smoke)
    JOINT_DIR="${JOINT_PRETRAIN_DIR}_smoke20"
    JOINT_STEPS=20
    JOINT_SAVE_EVERY=20
    JOINT_LOG_EVERY=5
    ;;
  joint_pretrain)
    JOINT_DIR="${JOINT_PRETRAIN_DIR}"
    JOINT_STEPS="${JOINT_PRETRAIN_STEPS:-3000}"
    JOINT_SAVE_EVERY="${SAVE_EVERY:-500}"
    JOINT_LOG_EVERY="${LOG_EVERY:-20}"
    ;;
  guard)
    if [[ -z "${VARIANT_WAS_SET}" ]]; then
      VARIANT="b0"
    fi
    OPTIMIZER_STEPS=0
    EVAL_EVERY=1
    SAVE_EVERY=1
    VAL_MAX_BATCHES=0
    WARMUP_STEPS=0
    TAG="guard"
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    OPTIMIZER_STEPS=20
    EVAL_EVERY=20
    SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    WARMUP_STEPS=5
    TAG="smoke20"
    VALIDATION_ARGS=()
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TAG="pilot${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  full)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TAG="full${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|joint_smoke|joint_pretrain|guard|smoke|pilot|full|summary" >&2
    exit 2
    ;;
esac

for required in "${SOURCE_A4_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}"; do
  test -s "${required}" || {
    echo "Missing required H5-B/C input: ${required}" >&2
    exit 1
  }
done

if [[ "${MODE}" == "joint_smoke" || "${MODE}" == "joint_pretrain" ]]; then
  mkdir -p "${JOINT_DIR}" "${LOG_ROOT}"
  if [[ -s "${JOINT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
    COMPLETED_STEPS="$(
      python - "${JOINT_DIR}/training_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(summary.get("completed_step", 0)))
PY
    )"
    if (( COMPLETED_STEPS >= JOINT_STEPS )); then
      echo "H5-C joint pretraining already completed: ${COMPLETED_STEPS}/${JOINT_STEPS} steps"
      exit 0
    fi
    echo "H5-C joint pretraining resumes: ${COMPLETED_STEPS} -> ${JOINT_STEPS} steps"
  fi
  RESUME_ARGS=()
  if [[ -s "${JOINT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
    RESUME_ARGS=(--resume "${JOINT_DIR}/latest_resume.pth")
  fi
  python -u script/standard_ML/pretrain_H5_C_joint_rgbsar.py \
    --source_a4_checkpoint "${SOURCE_A4_CKPT}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --sar_stats_path "${SAR_STATS_JSON}" \
    --output_dir "${JOINT_DIR}" \
    --device cuda:0 \
    --crop_size 640 \
    --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
    --optimizer_steps "${JOINT_STEPS}" \
    --save_every "${JOINT_SAVE_EVERY}" \
    --log_every "${JOINT_LOG_EVERY}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --learning_rate "${JOINT_PRETRAIN_LR:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --mask_ratio "${JOINT_MASK_RATIO:-0.60}" \
    --changed_margin "${CHANGED_MARGIN:-0.20}" \
    --amp "${AMP:-bf16}" \
    --seed "${SEED}" \
    "${RESUME_ARGS[@]}"
  exit 0
fi

case "${VARIANT}" in
  b0|b1|c1) ;;
  *)
    echo "Unknown H5-B/C variant '${VARIANT}'; choose b0|b1|c1" >&2
    exit 2
    ;;
esac

for required in "${DAMAGE_PRIORS_JSON}" "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing required supervised input: ${required}" >&2
    exit 1
  }
done

EXTRA_ARGS=()
if [[ "${VARIANT}" == "c1" ]]; then
  test -s "${JOINT_PRETRAIN_CKPT}" || {
    echo "Missing H5-C joint adapters: ${JOINT_PRETRAIN_CKPT}" >&2
    echo "Run '$0 joint_pretrain' first." >&2
    exit 1
  }
  EXTRA_ARGS+=(--joint_pretrained_checkpoint "${JOINT_PRETRAIN_CKPT}")
fi

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H5_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H5-B/C experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi

RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
SECONDARY_ARGS=()
if [[ "${SAVE_SECONDARY_BEST:-0}" == "1" ]]; then
  SECONDARY_ARGS=(--save_secondary_best)
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "H4-A4 source     : ${SOURCE_A4_CKPT}"
echo "B sparse fusion  : $([[ ${VARIANT} == b1 ]] && echo 'stage3+stage4, building only' || echo 'disabled')"
echo "C joint adapters : $([[ ${VARIANT} == c1 ]] && echo "${JOINT_PRETRAIN_CKPT}" || echo 'disabled')"
echo "Hard query decode: DISABLED"
echo "New sup. losses  : NONE"
echo "RGB/Localization : FROZEN / FROZEN"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_H5_BC.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${SOURCE_A4_CKPT}" \
  --damage_priors_json "${DAMAGE_PRIORS_JSON}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --new_module_warmup_steps "${WARMUP_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --core_learning_rate "${CORE_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --amp "${AMP:-bf16}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "${SECONDARY_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  python - "${SOURCE_A4_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json
import sys
import torch

source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
expected = source.get("metrics", source.get("best_metrics", {}))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 5e-5:
        raise SystemExit(f"H5 B0 guard mismatch for {key}")
print("H5 B0 full-validation guard passed.")
PY
fi
