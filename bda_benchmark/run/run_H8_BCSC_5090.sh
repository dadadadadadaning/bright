#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-c1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H8_BCSC}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H8_BCSC}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H8_BCSC_5090.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H8_BCSC.py \
      script/standard_ML/train_H8_BCSC.py \
      script/standard_ML/eval_H8_BCSC.py \
      script/standard_ML/audit_H8_BCSC_prototype_oracle.py
    python -m unittest -v tests.test_h8_bcsc
    echo "H8-BCSC preflight passed."
    exit 0
    ;;
  oracle_val)
    H8_CKPT="${H8_CKPT:-}"
    test -n "${H8_CKPT}" || {
      echo "oracle_val requires H8_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    for required in "${H8_CKPT}" "${SAR_STATS_JSON}" "${VAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing H8 oracle input: ${required}" >&2
        exit 1
      }
    done
    ORACLE_OUTPUT_JSON="${ORACLE_OUTPUT_JSON:-${H8_CKPT%/*}/prototype_oracle_val.json}"
    python -u script/standard_ML/audit_H8_BCSC_prototype_oracle.py \
      --checkpoint "${H8_CKPT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${ORACLE_OUTPUT_JSON}" \
      --blend_values ${ORACLE_BLEND_VALUES:-0.25 0.50 1.00} \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  guard)
    VARIANT="c0"
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; WARMUP_STEPS=0; TAG="guard"
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    WARMUP_STEPS="${WARMUP_STEPS:-5}"; TAG="smoke20"
    VALIDATION_ARGS=()
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-3000}"
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
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    H8_CKPT="${H8_CKPT:-}"
    test -n "${H8_CKPT}" || {
      echo "${MODE} requires H8_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    for required in "${H8_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing H8 evaluation input: ${required}" >&2
        exit 1
      }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${H8_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    python -u script/standard_ML/eval_H8_BCSC.py \
      --checkpoint "${H8_CKPT}" \
      --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|full|eval_val|eval_test|oracle_val" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in c0|c1|c2) ;; *) echo "Unknown H8 variant: ${VARIANT}" >&2; exit 2;; esac
for required in \
  "${H6_A2_CKPT}" "${SAR_STATS_JSON}" "${DAMAGE_PRIORS_JSON}" \
  "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required H8 input: ${required}" >&2; exit 1; }
done

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H8_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H8 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Locked H6-A2     : ${H6_A2_CKPT}"
echo "BCSC             : $([[ ${VARIANT} == c0 ]] && echo disabled || echo stage3+stage4)"
echo "Reliability gate : $([[ ${VARIANT} == c2 ]] && echo bounded-low-open || echo disabled)"
echo "Teacher/KD       : NONE / NONE"
echo "Test selection   : NEVER during development"
echo "RGB/Loc/BN       : FROZEN / FROZEN / FROZEN"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_H8_BCSC.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${H6_A2_CKPT}" \
  --damage_priors_json "${DAMAGE_PRIORS_JSON}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --dataset_variant h3 \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --new_module_warmup_steps "${WARMUP_STEPS}" \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --core_learning_rate "${CORE_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --amp "${AMP:-bf16}" --seed "${SEED}" \
  "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  python - "${H6_A2_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
expected = source.get("metrics", source.get("best_metrics", {}))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 5e-5:
        raise SystemExit(f"H8 C0 guard mismatch for {key}")
print("H8-C0 exact H6-A2 full-validation guard passed.")
PY
fi
