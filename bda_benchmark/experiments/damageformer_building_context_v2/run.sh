#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-full_joint}"

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

OLD_OUTPUT_ROOT="${OLD_OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_CONTEXT_V2}"
SEED="${SEED:-3407}"

case "${VARIANT}" in
  full_joint|bir_on_frozen_c5|bir_shuffle_on_frozen_c5|rgb_only_building|raw_sar_building|no_conf_building|shifted_sar_building|wrong_building_joint)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

stage_values() {
  case "$1" in
    smoke)   echo "smoke20 20 10 8" ;;
    probe)   echo "probe5000 5000 500 0" ;;
    confirm) echo "confirm20000 20000 1000 0" ;;
    *) echo "Unknown phase: $1" >&2; exit 2 ;;
  esac
}

preflight() {
  echo "[1/7] Python package"
  python -c "import bda_benchmark; print('bda_benchmark:', bda_benchmark.__file__)"

  echo "[2/7] Old C5 imports"
  python -c "from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel; print('old model import OK')"
  python -c "from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool; print('old association import OK')"

  echo "[3/7] Source syntax"
  python -m py_compile \
    experiments/damageformer_building_context_v2/variants.py \
    experiments/damageformer_building_context_v2/building_ops.py \
    experiments/damageformer_building_context_v2/core.py \
    experiments/damageformer_building_context_v2/model.py \
    experiments/damageformer_building_context_v2/train.py \
    experiments/damageformer_building_context_v2/summarize.py \
    experiments/damageformer_building_context_v2/test_contract.py

  echo "[4/7] Unit tests, including BF16"
  python -m unittest experiments.damageformer_building_context_v2.test_contract -v

  echo "[5/7] Checkpoint"
  test -s "${A1_CKPT}" || { echo "Missing A1: ${A1_CKPT}" >&2; exit 3; }
  ls -lh "${A1_CKPT}"

  echo "[6/7] Train/Val split"
  test -s "${TRAIN_LIST}" || { echo "Missing train split" >&2; exit 3; }
  test -s "${VAL_LIST}" || { echo "Missing val split" >&2; exit 3; }
  python - <<PY
from pathlib import Path
for p in [Path("${TRAIN_LIST}"), Path("${VAL_LIST}")]:
    rows=[x.strip() for x in p.read_text().splitlines() if x.strip()]
    print(p, len(rows))
PY

  echo "[7/7] CUDA"
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

  echo "PREFLIGHT PASSED | TEST ACCESS FORBIDDEN"
}

train_one() {
  local phase="$1"
  local tag steps eval_every val_max out c5_ckpt
  local -a extra=()
  local -a resume=()

  read -r tag steps eval_every val_max <<< "$(stage_values "${phase}")"
  out="${OUTPUT_ROOT}/${VARIANT}_${tag}_seed${SEED}"

  if [[ -s "${out}/training_summary.json" ]]; then
    echo "Completed, skip: ${out}"
    return
  fi

  mkdir -p "${out}"
  [[ -s "${out}/latest.pth" ]] && resume+=(--resume "${out}/latest.pth")

  if [[ "${VARIANT}" == "bir_on_frozen_c5" || "${VARIANT}" == "bir_shuffle_on_frozen_c5" ]]; then
    c5_ckpt="${C5_CKPT:-${OLD_OUTPUT_ROOT}/c5_aligned_assoc_${tag}_seed${SEED}/best_val_miou.pth}"
    test -s "${c5_ckpt}" || {
      echo "Missing old C5 checkpoint: ${c5_ckpt}" >&2
      echo "Run the old c5_aligned_assoc ${phase} first." >&2
      exit 4
    }
    extra+=(--c5_ckpt "${c5_ckpt}")
  fi

  python -u -m bda_benchmark.experiments.damageformer_building_context_v2.train \
    --variant "${VARIANT}" \
    --dataset_path "${DATASET_PATH}" \
    --train_list "${TRAIN_LIST}" \
    --val_list "${VAL_LIST}" \
    --a1_ckpt "${A1_CKPT}" \
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
    --purity_threshold "${PURITY_THRESHOLD:-0.60}" \
    --device "${DEVICE:-cuda:0}" \
    --amp_dtype "${AMP_DTYPE:-bf16}" \
    --seed "${SEED}" \
    "${extra[@]}" \
    "${resume[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke|probe|confirm) preflight; train_one "${MODE}" ;;
  *)
    echo "Usage: $0 {preflight|smoke|probe|confirm} [variant]" >&2
    exit 2
    ;;
esac
