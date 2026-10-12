#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-t1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H11_SATR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H11_SATR}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
SARMAE_CKPT="${SARMAE_CKPT:-}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${VARIANT}" in
  t2|t2e) TRAINABLE_SARMAE_TAIL=1 ;;
  *) TRAINABLE_SARMAE_TAIL=0 ;;
esac
case "${VARIANT}" in
  t1e|t2e) EXACT_SARMAE_INPUT=1; DATASET_VARIANT=h4 ;;
  *) EXACT_SARMAE_INPUT=0; DATASET_VARIANT=h3 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H11_SATR_5090.sh
    bash -n run/run_H11_SATR_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H11_SATR.py \
      script/standard_ML/train_H11_SATR.py \
      script/standard_ML/eval_H11_SATR.py \
      script/standard_ML/check_H11_SARMAE_checkpoint.py
    python -m unittest -v tests.test_h11_satr
    echo "H11-SATR preflight passed."
    exit 0
    ;;
  check_sarmae)
    test -n "${SARMAE_CKPT}" || {
      echo "check_sarmae requires SARMAE_CKPT=/path/to/checkpoint.pth" >&2; exit 1;
    }
    test -s "${SARMAE_CKPT}" || { echo "Missing SARMAE checkpoint: ${SARMAE_CKPT}" >&2; exit 1; }
    python -u script/standard_ML/check_H11_SARMAE_checkpoint.py \
      --checkpoint "${SARMAE_CKPT}" \
      --output_json "${SARMAE_AUDIT_JSON:-${SARMAE_CKPT}.h11_audit.json}"
    exit 0
    ;;
  guard)
    [[ "${VARIANT}" == "t0" ]] || { echo "Guard requires VARIANT=t0" >&2; exit 2; }
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; TAG="guard"; WARMUP_STEPS=0
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    [[ "${VARIANT}" == "t1" || "${VARIANT}" == "t2" || "${VARIANT}" == "t1e" || "${VARIANT}" == "t2e" ]] || {
      echo "Smoke supports t1|t2|t1e|t2e" >&2; exit 2;
    }
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"; TAG="smoke20"
    WARMUP_STEPS=$([[ "${TRAINABLE_SARMAE_TAIL}" == "1" ]] && echo 5 || echo 0)
    VALIDATION_ARGS=()
    ;;
  screen)
    [[ "${VARIANT}" == "t1" || "${VARIANT}" == "t2" || "${VARIANT}" == "t1e" || "${VARIANT}" == "t2e" ]] || {
      echo "Screen supports t1|t2|t1e|t2e" >&2; exit 2;
    }
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}"
    EVAL_EVERY="${EVAL_EVERY:-500}"; SAVE_EVERY="${SAVE_EVERY:-1500}"
    VAL_MAX_BATCHES=0; TAG="screen${OPTIMIZER_STEPS}"
    WARMUP_STEPS=$([[ "${TRAINABLE_SARMAE_TAIL}" == "1" ]] && echo "${WARMUP_STEPS:-500}" || echo 0)
    VALIDATION_ARGS=()
    ;;
  pilot)
    [[ "${VARIANT}" == "t1" || "${VARIANT}" == "t2" || "${VARIANT}" == "t1e" || "${VARIANT}" == "t2e" ]] || {
      echo "Pilot supports t1|t2|t1e|t2e" >&2; exit 2;
    }
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-5000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"; SAVE_EVERY="${SAVE_EVERY:-5000}"
    VAL_MAX_BATCHES=0; TAG="controlled${OPTIMIZER_STEPS}"
    WARMUP_STEPS=$([[ "${TRAINABLE_SARMAE_TAIL}" == "1" ]] && echo "${WARMUP_STEPS:-500}" || echo 0)
    VALIDATION_ARGS=()
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    H11_CKPT="${H11_CKPT:-}"
    test -n "${H11_CKPT}" || {
      echo "${MODE} requires H11_CKPT=/path/to/best_val_miou.pth" >&2; exit 1;
    }
    for required in "${H11_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || { echo "Missing H11 evaluation input: ${required}" >&2; exit 1; }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${H11_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    python -u script/standard_ML/eval_H11_SATR.py \
      --checkpoint "${H11_CKPT}" --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|check_sarmae|guard|smoke|screen|pilot|eval_val|eval_test" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in t0|t1|t2|t1e|t2e) ;; *) echo "Unknown H11 variant: ${VARIANT}" >&2; exit 2;; esac
for required in "${H6_A2_CKPT}" "${SAR_STATS_JSON}" "${DAMAGE_PRIORS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required H11 input: ${required}" >&2; exit 1; }
done
SARMAE_ARGS=()
if [[ "${VARIANT}" != "t0" ]]; then
  test -n "${SARMAE_CKPT}" || {
    echo "H11-${VARIANT^^} requires SARMAE_CKPT=/path/to/official_encoder.pth" >&2; exit 1;
  }
  test -s "${SARMAE_CKPT}" || { echo "Missing SARMAE checkpoint: ${SARMAE_CKPT}" >&2; exit 1; }
  SARMAE_ARGS=(--sar_pretrained_checkpoint "${SARMAE_CKPT}")
fi

if [[ -z "${EXP_NAME}" ]]; then EXP_NAME="H11_${VARIANT^^}_${TAG}_seed${SEED}"; fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H11 experiment already exists; skipped: ${OUTPUT_DIR}"; exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Locked H6-A2     : ${H6_A2_CKPT}"
echo "SARMAE source    : $([[ ${VARIANT} == t0 ]] && echo disabled || echo "${SARMAE_CKPT}")"
echo "SARMAE input     : $([[ ${EXACT_SARMAE_INPUT} == 1 ]] && echo official-3ch-ImageNet-from-raw-display || echo naive-1ch-train-stats)"
echo "Transfer stages  : $([[ ${VARIANT} == t0 ]] && echo disabled || echo stage3+stage4)"
echo "MAE fine-tuning  : $([[ ${TRAINABLE_SARMAE_TAIL} == 1 ]] && echo blocks10-11-after-${WARMUP_STEPS} || echo frozen)"
echo "Dataset contract : ${DATASET_VARIANT}"
echo "Native A2/RGB    : FROZEN / FROZEN"
echo "New losses       : NONE; unchanged H6-A2 objective"
echo "Severity head    : NOT ADDED"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-1} * ${ACCUMULATION_STEPS:-16} ))"
echo "================================================================================"

python -u script/standard_ML/train_H11_SATR.py \
  --variant "${VARIANT}" --source_h3_checkpoint "${H6_A2_CKPT}" \
  "${SARMAE_ARGS[@]}" \
  --damage_priors_json "${DAMAGE_PRIORS_JSON}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant "${DATASET_VARIANT}" \
  --output_dir "${OUTPUT_DIR}" --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-16}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" --new_module_warmup_steps "${WARMUP_STEPS}" \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" --core_learning_rate 0 \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" --amp "${AMP:-bf16}" --seed "${SEED}" \
  --save_secondary_best "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  python - "${H6_A2_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = source.get("metrics", source.get("best_metrics", {}))
observed = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"H11 T0 exact H6-A2 guard failed for {key}: {delta}")
print("H11-T0 exact H6-A2 guard passed.")
PY
fi
