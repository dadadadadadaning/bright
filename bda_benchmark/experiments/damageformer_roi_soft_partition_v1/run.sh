#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_ROI_SOFT_PARTITION_V1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_ROI_SOFT_PARTITION_V1}"
SEED="${SEED:-3407}"

stage_values() {
  case "$1" in
    smoke) echo "smoke20 20 10 8" ;;
    train) echo "train10000 10000 500 0" ;;
    *) echo "Unknown phase: $1" >&2; exit 2 ;;
  esac
}

preflight() {
  echo "============================================================"
  echo "[1/8] Working directory / PYTHONPATH"
  echo "============================================================"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "REPO_ROOT=${REPO_ROOT}"
  pwd

  echo
  echo "============================================================"
  echo "[2/8] Required old partition implementation"
  echo "============================================================"
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/model.py
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/association.py
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/losses.py
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/train.py
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/core.py
  python -c "from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel; print('OLD PARTITION MODEL IMPORT: OK')"

  echo
  echo "============================================================"
  echo "[3/8] New experiment syntax"
  echo "============================================================"
  python -m py_compile \
    experiments/damageformer_roi_soft_partition_v1/model.py \
    experiments/damageformer_roi_soft_partition_v1/train.py \
    experiments/damageformer_roi_soft_partition_v1/test_contract.py \
    experiments/damageformer_roi_soft_partition_v1/inspect_result.py
  echo "PY_COMPILE: OK"

  echo
  echo "============================================================"
  echo "[4/8] Fixed method contract"
  echo "============================================================"
  python -m unittest experiments.damageformer_roi_soft_partition_v1.test_contract -v

  echo
  echo "============================================================"
  echo "[5/8] A1 parent checkpoint"
  echo "============================================================"
  test -s "${A1_CKPT}" || { echo "Missing A1 checkpoint: ${A1_CKPT}" >&2; exit 3; }
  ls -lh "${A1_CKPT}"

  echo
  echo "============================================================"
  echo "[6/8] Train / Validation only"
  echo "============================================================"
  test -s "${TRAIN_LIST}" || { echo "Missing TRAIN_LIST: ${TRAIN_LIST}" >&2; exit 3; }
  test -s "${VAL_LIST}" || { echo "Missing VAL_LIST: ${VAL_LIST}" >&2; exit 3; }

  python - <<PY
from pathlib import Path
for p in [Path(r"${TRAIN_LIST}"), Path(r"${VAL_LIST}")]:
    if "test" in p.name.lower():
        raise SystemExit(f"TEST ACCESS FORBIDDEN: {p}")
    rows = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"{p}: {len(rows)} records")
    if not rows:
        raise SystemExit(f"EMPTY SPLIT: {p}")
PY

  echo
  echo "============================================================"
  echo "[7/8] CUDA"
  echo "============================================================"
  python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

  echo
  echo "============================================================"
  echo "[8/8] Protocol"
  echo "============================================================"
  python - <<'PY'
from bda_benchmark.experiments.damageformer_roi_soft_partition_v1.model import FIXED_VARIANT
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.variants import get_variant
s = get_variant(FIXED_VARIANT)
print("Top-level module 1: A1 soft RGB-building ROI")
print("Top-level module 2: cross-modal damage-aware soft partition")
print("Fixed variant      :", FIXED_VARIANT)
print("Partition          :", s.partition)
print("SAR source         :", s.sar_source)
print("Use confidence     :", s.use_alignment_confidence)
print("Use SLICO          :", s.partition == "slico")
print("Building reasoning : False")
print("Graph              : False")
print("MIL                : False")
print("Memory             : False")
print("Test access        : FORBIDDEN")
PY
  echo
  echo "PREFLIGHT PASSED"
}

train_one() {
  local phase="$1"
  local tag steps eval_every val_max out
  read -r tag steps eval_every val_max <<< "$(stage_values "${phase}")"

  out="${OUTPUT_ROOT}/roi_soft_partition_${tag}_seed${SEED}"
  mkdir -p "${out}" "${LOG_ROOT}"

  local -a resume=()

  if [[ -s "${out}/training_summary.json" ]]; then
    echo "Completed output already exists:"
    echo "${out}/training_summary.json"
    return
  fi

  if [[ -s "${out}/latest.pth" ]]; then
    echo "Resume checkpoint found: ${out}/latest.pth"
    resume+=(--resume "${out}/latest.pth")
  fi

  python -u -m bda_benchmark.experiments.damageformer_roi_soft_partition_v1.train \
    --dataset_path "${DATASET_PATH}" \
    --train_list "${TRAIN_LIST}" \
    --val_list "${VAL_LIST}" \
    --parent_ckpt "${A1_CKPT}" \
    --output_dir "${out}" \
    --optimizer_steps "${steps}" \
    --eval_every "${eval_every}" \
    --save_every "${eval_every}" \
    --val_max_batches "${val_max}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --plugin_lr "${PLUGIN_LR:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --region_stride "${REGION_STRIDE:-8}" \
    --device "${DEVICE:-cuda:0}" \
    --amp_dtype "${AMP_DTYPE:-bf16}" \
    --seed "${SEED}" \
    "${resume[@]}"
}

show_status() {
  echo "=== processes ==="
  pgrep -af 'damageformer_roi_soft_partition_v1.train' || true
  echo
  echo "=== summaries ==="
  find "${OUTPUT_ROOT}" -maxdepth 2 -name training_summary.json -print 2>/dev/null || true
  echo
  echo "=== latest checkpoints ==="
  find "${OUTPUT_ROOT}" -maxdepth 2 -name latest.pth -print 2>/dev/null || true
}

show_summary() {
  local out="${OUTPUT_ROOT}/roi_soft_partition_train10000_seed${SEED}"
  python -m bda_benchmark.experiments.damageformer_roi_soft_partition_v1.inspect_result \
    --output_dir "${out}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) preflight; train_one smoke ;;
  train) preflight; train_one train ;;
  status) show_status ;;
  summary) show_summary ;;
  *)
    echo "Usage: $0 {preflight|smoke|train|status|summary}" >&2
    exit 2
    ;;
esac
