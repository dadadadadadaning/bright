#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT_WAS_SET="${VARIANT+x}"
VARIANT="${VARIANT:-q1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H5_BIQ}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
SOURCE_A4_CKPT="${SOURCE_A4_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/H4_A4_pilot3000_seed3407/best_val_miou.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H5_BIQ_5090.sh
    bash -n run/run_H5_BIQ_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H5_BIQ.py \
      script/standard_ML/train_DamageFormer_RGBSAR_H5_BIQ.py \
      script/standard_ML/summarize_H5_BIQ_results.py
    python -m unittest -v tests.test_damageformer_rgbsar_h5_biq
    echo "H5 BIQ preflight passed."
    exit 0
    ;;
  summary)
    python -u script/standard_ML/summarize_H5_BIQ_results.py \
      --output_root "${OUTPUT_ROOT}" \
      --baseline_json "${BASELINE_JSON:-}" \
      --output_json "${OUTPUT_ROOT}/H5_BIQ_validation_comparison.json"
    exit 0
    ;;
  guard)
    if [[ -z "${VARIANT_WAS_SET}" ]]; then
      VARIANT="q0"
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
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    TAG="pilot${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  full)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    TAG="full${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|full|summary" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  q0|q1|q2) ;;
  *)
    echo "Unknown H5 variant '${VARIANT}'; choose q0|q1|q2" >&2
    exit 2
    ;;
esac

for required in \
  "${SOURCE_A4_CKPT}" \
  "${SAR_STATS_JSON}" \
  "${DAMAGE_PRIORS_JSON}" \
  "${TRAIN_LIST}" \
  "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing required H5 input: ${required}" >&2
    exit 1
  }
done

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H5_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H5 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi

RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
SECONDARY_ARGS=()
if [[ "${SAVE_SECONDARY_BEST:-1}" == "1" ]]; then
  SECONDARY_ARGS=(--save_secondary_best)
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "H4-A4 source     : ${SOURCE_A4_CKPT}"
echo "Prediction unit  : building instance query"
echo "Sparse fusion    : $([[ ${VARIANT} == q2 ]] && echo 'stage3+stage4' || echo 'disabled')"
echo "New losses       : NONE"
echo "RGB/Localization : FROZEN / FROZEN"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/16"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_H5_BIQ.py \
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
        raise SystemExit(f"H5 Q0 guard mismatch for {key}")
print("H5 Q0 full-validation guard passed.")
PY
fi
