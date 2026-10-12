#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-m1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
A2_CKPT="${A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/A2_DCSM}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/A2_DCSM}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_A2_DCSM_5090.sh
    bash -n run/run_A2_DCSM_sequence.sh
    python -m py_compile \
      dataset/make_data_loader_heterogeneous_rgbsar.py \
      model/DamageFormer_RGBSAR_A2_DCSM.py \
      script/standard_ML/pretrain_A2_DCSM.py \
      script/standard_ML/train_A2_DCSM.py \
      script/standard_ML/eval_A2_DCSM.py
    python -c \
      "from bda_benchmark.script.standard_ML.train_A2_DCSM import validate_dataset_interface; validate_dataset_interface(); print('A2-DCSM dataset interface: OK')"
    python -m unittest -v tests.test_a2_dcsm
    echo "A2-DCSM preflight passed."
    exit 0
    ;;
  eval_val|eval_test)
    DCSM_CKPT="${DCSM_CKPT:-}"
    test -s "${DCSM_CKPT}" || {
      echo "${MODE} requires DCSM_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    if [[ "${MODE}" == "eval_test" ]]; then
      SPLIT=test; LIST="${TEST_LIST}"
    else
      SPLIT=val; LIST="${VAL_LIST}"
    fi
    OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${DCSM_CKPT%/*}/independent_${SPLIT}.json}"
    python -u script/standard_ML/eval_A2_DCSM.py \
      --checkpoint "${DCSM_CKPT}" \
      --split "${SPLIT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
esac

case "${VARIANT}" in m0|m1|m2|m3) ;; *) echo "Unknown variant ${VARIANT}" >&2; exit 2;; esac
for required in "${A2_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required A2-DCSM input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

PRETRAIN_STEPS="${PRETRAIN_STEPS:-5000}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${OUTPUT_ROOT}/A2_${VARIANT^^}_pretrain${PRETRAIN_STEPS}_seed${SEED}}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${PRETRAIN_DIR}/pathway_final.pth}"

case "${MODE}" in
  pretrain_smoke|pretrain)
    [[ "${VARIANT}" != "m0" ]] || {
      echo "M0 is the locked control and has no pretraining stage" >&2; exit 2;
    }
    if [[ "${MODE}" == "pretrain_smoke" ]]; then
      STEPS=20
      PRETRAIN_DIR="${OUTPUT_ROOT}/A2_${VARIANT^^}_pretrain_smoke20_seed${SEED}"
      SAVE_EVERY=20
    else
      STEPS="${PRETRAIN_STEPS}"
      SAVE_EVERY="${SAVE_EVERY:-1000}"
    fi
    mkdir -p "${PRETRAIN_DIR}"
    if [[ -s "${PRETRAIN_DIR}/pathway_final.pth" && "${FORCE_RERUN}" != 1 ]]; then
      echo "Completed DCSM pathway already exists; skipped: ${PRETRAIN_DIR}/pathway_final.pth"
      exit 0
    fi
    RESUME_ARGS=()
    if [[ -s "${PRETRAIN_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != 1 ]]; then
      RESUME_ARGS=(--resume "${PRETRAIN_DIR}/latest_resume.pth")
    fi
    python -u script/standard_ML/pretrain_A2_DCSM.py \
      --variant "${VARIANT}" \
      --source_a2_checkpoint "${A2_CKPT}" \
      --train_dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_dir "${PRETRAIN_DIR}" \
      --device cuda:0 --crop_size 640 \
      --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
      --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
      --optimizer_steps "${STEPS}" \
      --learning_rate "${PRETRAIN_LR:-1e-4}" \
      --weight_decay "${WEIGHT_DECAY:-5e-3}" \
      --mask_ratio "${MASK_RATIO:-0.60}" \
      --rgb_anchor_strength "${RGB_ANCHOR_STRENGTH:-0.25}" \
      --focus_probability "${FOCUS_PROBABILITY:-0.75}" \
      --save_every "${SAVE_EVERY}" --log_every "${LOG_EVERY:-20}" \
      --num_workers "${NUM_WORKERS:-8}" --amp "${AMP:-bf16}" --seed "${SEED}" \
      "${RESUME_ARGS[@]}"
    exit 0
    ;;
  guard)
    VARIANT=m0; STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; TAG=guard
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20; TAG=smoke20
    VALIDATION_ARGS=(--val_max_batches "${VAL_MAX_BATCHES:-2}")
    ;;
  pilot)
    STEPS="${OPTIMIZER_STEPS:-3000}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"; TAG=pilot${STEPS}
    VALIDATION_ARGS=(--val_max_batches 0)
    ;;
  full)
    STEPS="${OPTIMIZER_STEPS:-8000}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"; TAG=full${STEPS}
    VALIDATION_ARGS=(--val_max_batches 0)
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|pretrain_smoke|pretrain|guard|smoke|pilot|full|eval_val|eval_test" >&2
    exit 2
    ;;
esac

if [[ "${VARIANT}" != m0 ]]; then
  test -s "${PRETRAIN_CKPT}" || {
    echo "Missing ${VARIANT^^} pathway checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
  }
fi

EXP_NAME="${EXP_NAME:-A2_${VARIANT^^}_${TAG}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != 1 ]]; then
  echo "Completed experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != 1 ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
PRETRAIN_ARGS=()
if [[ "${VARIANT}" != m0 ]]; then
  PRETRAIN_ARGS=(--sar_pretrained_checkpoint "${PRETRAIN_CKPT}")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Locked H6-A2     : ${A2_CKPT}"
echo "DCSM pathway     : $([[ ${VARIANT} == m0 ]] && echo disabled || echo "${PRETRAIN_CKPT}")"
echo "Sampling         : event-balanced + damage-focused crops"
echo "Supervised loss  : unchanged A2 losses; no pretraining loss"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_A2_DCSM.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${A2_CKPT}" \
  "${PRETRAIN_ARGS[@]}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant h3 \
  --output_dir "${OUTPUT_DIR}" --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${STEPS}" --new_module_warmup_steps 0 \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --core_learning_rate "${CORE_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" --amp "${AMP:-bf16}" --seed "${SEED}" \
  "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${MODE}" == guard ]]; then
  python - "${A2_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
expected = source.get("metrics", source.get("best_metrics", {}))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 5e-5: raise SystemExit(f"A2-M0 guard mismatch for {key}")
print("A2-M0 exact H6-A2 guard passed.")
PY
fi
