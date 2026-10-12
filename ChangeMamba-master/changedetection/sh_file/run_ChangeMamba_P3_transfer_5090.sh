#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${VARIANT:-cm_p2}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master/ChangeMamba-master}"
CHANGEDIR="${PROJECT_ROOT}/changedetection"
DATA_ROOT="${DATA_ROOT:-/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset}"
LIST_ROOT="${LIST_ROOT:-${CHANGEDIR}/datasets/data_name_list/BRIGHT}"
CFG_PATH="${CFG_PATH:-${CHANGEDIR}/configs/vssm1/vssm_tiny_224_0229flex.yaml}"
SAR_STATS="${SAR_STATS:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
BASELINE_CKPT="${BASELINE_CKPT:-${PROJECT_ROOT}/results/checkpoints_B0_full/BRIGHT/ChangeMamba-MMBDA_B0_cat_1783259203.4570115/best_model.pth}"
PARENT_CKPT="${PARENT_CKPT:-${BASELINE_CKPT}}"
EXPECTED_PARENT_MIOU="${EXPECTED_PARENT_MIOU:-64.2585}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/p3_transfer}"
SEED="${SEED:-3407}"
ROI_FLOOR="${ROI_FLOOR:-0.10}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${VARIANT}_3000_seed${SEED}}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${OUTPUT_ROOT}/${VARIANT}_smoke20_seed${SEED}}"

case "${VARIANT}" in
  cm_p2|cm_p3) ;;
  *) echo "VARIANT must be cm_p2 or cm_p3" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${CHANGEDIR}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    changedetection/experiments/p3_transfer/model.py \
    changedetection/experiments/p3_transfer/trainer.py \
    changedetection/script/train_ChangeMamba_p3_transfer.py \
    changedetection/script/eval_ChangeMamba_p3_transfer.py
  python -m unittest changedetection.tests.test_changemamba_p3_transfer -v
  # Unit tests exercise the model mathematics.  Import the real trainer as a
  # separate dependency audit so missing runtime packages (for example yacs)
  # fail during preflight rather than after a background job is launched.
  python -c 'from changedetection.experiments.p3_transfer.trainer import P3TransferChangeMambaTrainer; print("P3-transfer trainer import: READY")'
  # VMamba deliberately swallows optional CUDA-extension import errors.  This
  # experiment is GPU-only, so audit the real kernels explicitly before a
  # background job reaches its first validation forward pass.
  python - <<'PY'
import importlib
import torch

required = (
    "selective_scan_cuda_core",
    "selective_scan_cuda_ndstate",
    "selective_scan_cuda_oflex",
)
failures = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}: READY | {module.__file__}")
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")

if failures:
    raise RuntimeError(
        "VMamba selective-scan CUDA extension is unavailable:\n  "
        + "\n  ".join(failures)
        + "\nRebuild kernels/selective_scan inside the active conda environment."
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the active Python environment")
print(
    "CUDA device: READY | "
    f"{torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}"
)
PY
  echo "ChangeMamba P3-transfer preflight passed."
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
  echo "${label}: READY | ${path}"
}

train_run() {
  local steps="$1"
  local eval_every="$2"
  local run_output="$3"
  echo "=========================================================================="
  echo "Variant              : ${VARIANT}"
  echo "Parent checkpoint    : ${PARENT_CKPT}"
  echo "Expected parent mIoU : ${EXPECTED_PARENT_MIOU}"
  echo "SAR statistics       : ${SAR_STATS}"
  echo "Output               : ${run_output}"
  echo "Steps / eval every   : ${steps} / ${eval_every}"
  echo "=========================================================================="
  require_file "parent checkpoint" "${PARENT_CKPT}"
  require_file "SAR statistics" "${SAR_STATS}"
  mkdir -p "${run_output}"
  python -u changedetection/script/train_ChangeMamba_p3_transfer.py \
    --cfg "${CFG_PATH}" \
    --variant "${VARIANT}" \
    --parent_checkpoint "${PARENT_CKPT}" \
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
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --roi_floor "${ROI_FLOOR}" \
    --expected_parent_miou "${EXPECTED_PARENT_MIOU}" \
    --baseline_tolerance_pp "${BASELINE_TOLERANCE_PP:-0.05}" \
    --seed "${SEED}"
}

eval_run() {
  local split="$1"
  local list_path="${LIST_ROOT}/val_set.txt"
  local extra=()
  local ckpt="${P3_TRANSFER_CKPT:-${TRAIN_OUTPUT_DIR}/best_model.pth}"
  if [[ "${split}" == "test" ]]; then
    if [[ "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Test locked. Set ALLOW_TEST=1 only after validation selection." >&2
      exit 1
    fi
    list_path="${LIST_ROOT}/test_set.txt"
    extra+=(--allow_test)
  fi
  test -s "${ckpt}"
  python -u changedetection/script/eval_ChangeMamba_p3_transfer.py \
    --cfg "${CFG_PATH}" \
    --variant "${VARIANT}" \
    --checkpoint "${ckpt}" \
    --sar_stats_path "${SAR_STATS}" \
    --dataset_path "${DATA_ROOT}" \
    --data_list_path "${list_path}" \
    --split "${split}" \
    --output_json "${TRAIN_OUTPUT_DIR}/independent_${split}.json" \
    --roi_floor "${ROI_FLOOR}" \
    --crop_size 512 \
    --batch_size 4 \
    --num_workers 2 \
    "${extra[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) preflight; train_run 20 20 "${SMOKE_OUTPUT_DIR}" ;;
  train3000) preflight; train_run 3000 500 "${TRAIN_OUTPUT_DIR}" ;;
  eval_val) eval_run val ;;
  eval_test) eval_run test ;;
  *) echo "Usage: $0 preflight|smoke|train3000|eval_val|eval_test" >&2; exit 2 ;;
esac
