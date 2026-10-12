#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-cm0_original}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CHANGEDIR="${PROJECT_ROOT}/changedetection"
DATA_ROOT="${DATA_ROOT:-/root/data/BRIGHT-master/bda_benchmark/dataset}"
LIST_ROOT="${LIST_ROOT:-${CHANGEDIR}/datasets/data_name_list/BRIGHT}"
CFG_PATH="${CFG_PATH:-${CHANGEDIR}/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
PRETRAIN="${PRETRAIN:-${PROJECT_ROOT}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth}"
INIT_CKPT="${INIT_CKPT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1}"
EVAL_ROOT="${EVAL_ROOT:-/root/data/BRIGHT-eval-results/CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1}"
STEPS="${STEPS:-40000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-3407}"
ROI_FLOOR="${ROI_FLOOR:-0.10}"
RUN_DIR="${OUTPUT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}"
EVAL_JSON="${EVAL_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}_val.json"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${CHANGEDIR}:${PYTHONPATH:-}"

case "${VARIANT}" in
  cm0_original|cm1_building_prior) ;;
  *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
  echo "${label}: READY | ${path}"
}

preflight() {
  require_file "config" "${CFG_PATH}"
  if [[ -n "${INIT_CKPT}" ]]; then
    require_file "full-model initial checkpoint" "${INIT_CKPT}"
  else
    require_file "encoder pretrain" "${PRETRAIN}"
  fi
  require_file "train list" "${LIST_ROOT}/train_set.txt"
  require_file "validation list" "${LIST_ROOT}/val_set.txt"
  python -m py_compile \
    changedetection/experiments/building_prior_ablation_v1/model.py \
    changedetection/experiments/building_prior_ablation_v1/trainer.py \
    changedetection/experiments/building_prior_ablation_v1/train.py \
    changedetection/experiments/building_prior_ablation_v1/evaluate.py \
    changedetection/experiments/building_prior_ablation_v1/summarize.py
  python -m unittest \
    changedetection.experiments.building_prior_ablation_v1.test_core -v
  python - <<'PY'
import importlib
import torch

for name in (
    "selective_scan_cuda_core",
    "selective_scan_cuda_ndstate",
    "selective_scan_cuda_oflex",
):
    module = importlib.import_module(name)
    print(f"{name}: READY | {module.__file__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("CUDA:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
  echo "Building-prior ablation V1 preflight passed. Test access: FORBIDDEN."
}

guard_new_run_dir() {
  if [[ -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ -z "${RESUME_CKPT:-}" ]]; then
      echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
      echo "Use a new OUTPUT_ROOT, or set RESUME_CKPT to continue latest.pth." >&2
      exit 1
    fi
  fi
}

train_variant() {
  guard_new_run_dir
  mkdir -p "${RUN_DIR}"
  local resume_args=()
  local init_args=()
  if [[ -n "${INIT_CKPT}" ]]; then
    init_args+=(--init_checkpoint "${INIT_CKPT}")
  else
    init_args+=(--encoder_pretrained_path "${PRETRAIN}")
  fi
  if [[ -n "${RESUME_CKPT:-}" ]]; then
    require_file "resume checkpoint" "${RESUME_CKPT}"
    resume_args+=(--resume_training_path "${RESUME_CKPT}")
  fi
  python -u -m \
    changedetection.experiments.building_prior_ablation_v1.train \
    --cfg "${CFG_PATH}" \
    --variant "${VARIANT}" \
    --output_dir "${RUN_DIR}" \
    --train_dataset_path "${DATA_ROOT}" \
    --train_data_list_path "${LIST_ROOT}/train_set.txt" \
    --val_dataset_path "${DATA_ROOT}" \
    --val_data_list_path "${LIST_ROOT}/val_set.txt" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --eval_num_workers "${EVAL_NUM_WORKERS:-2}" \
    --crop_size 512 \
    --max_iters "${STEPS}" \
    --eval_every "${EVAL_EVERY}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --roi_floor "${ROI_FLOOR}" \
    --seed "${SEED}" \
    "${init_args[@]}" \
    "${resume_args[@]}"
}

eval_variant() {
  local checkpoint="${CHECKPOINT:-${RUN_DIR}/best_model.pth}"
  require_file "checkpoint" "${checkpoint}"
  mkdir -p "${EVAL_ROOT}"
  python -u -m \
    changedetection.experiments.building_prior_ablation_v1.evaluate \
    --cfg "${CFG_PATH}" \
    --variant "${VARIANT}" \
    --checkpoint "${checkpoint}" \
    --dataset_path "${DATA_ROOT}" \
    --data_list_path "${LIST_ROOT}/val_set.txt" \
    --output_json "${EVAL_JSON}" \
    --crop_size 512 \
    --batch_size "${EVAL_BATCH_SIZE:-4}" \
    --num_workers "${EVAL_NUM_WORKERS:-2}" \
    --roi_floor "${ROI_FLOOR}" \
    --seed "${SEED}"
}

summarize_pair() {
  mkdir -p "${EVAL_ROOT}"
  python -u -m \
    changedetection.experiments.building_prior_ablation_v1.summarize \
    --cm0 "${EVAL_ROOT}/cm0_original_step${STEPS}_seed${SEED}_val.json" \
    --cm1 "${EVAL_ROOT}/cm1_building_prior_step${STEPS}_seed${SEED}_val.json" \
    --output "${EVAL_ROOT}/summary_step${STEPS}_seed${SEED}.json"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    STEPS=20 EVAL_EVERY=20 OUTPUT_ROOT="${OUTPUT_ROOT}/smoke" \
      bash "${SCRIPT_DIR}/run.sh" train "${VARIANT}"
    ;;
  train) preflight; train_variant ;;
  eval_val) eval_variant ;;
  summarize) summarize_pair ;;
  *)
    echo "Usage: $0 preflight|smoke|train|eval_val|summarize [variant]" >&2
    exit 2
    ;;
esac
