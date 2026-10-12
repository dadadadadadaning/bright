#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-c5_aligned_assoc}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${VARIANT}" in
  c0_a1|c1_lca|c2_slico|c3_rgb_assoc|c4_raw_sar_assoc|c5_aligned_assoc|c6_no_confidence|c7_shuffled_sar|c8_damage_hint|c9_boundary) ;;
  *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1}"
SEED="${SEED:-3407}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    experiments/damageformer_a1_cross_modal_superpixel_v1/variants.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/association.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/losses.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/model.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/train.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/audit.py \
    experiments/damageformer_a1_cross_modal_superpixel_v1/summarize.py
  python -m unittest experiments.damageformer_a1_cross_modal_superpixel_v1.test_contract -v
  test -s "${A1_CKPT}" || { echo "Missing A1 checkpoint: ${A1_CKPT}" >&2; exit 3; }
  test -s "${TRAIN_LIST}"; test -s "${VAL_LIST}"
  echo "A1 cross-modal superpixel V1 preflight passed. Test access: FORBIDDEN."
}

stage_values() {
  case "$1" in
    smoke) echo "smoke20 20 10 8" ;;
    probe) echo "probe5000 5000 500 0" ;;
    confirm) echo "confirm20000 20000 1000 0" ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
  esac
}

train_one() {
  local phase="$1" tag steps eval_every val_max out resume=()
  [[ "${VARIANT}" != "c0_a1" ]] || { echo "c0_a1 is audit-only" >&2; exit 2; }
  read -r tag steps eval_every val_max <<<"$(stage_values "${phase}")"
  out="${OUTPUT_ROOT}/${VARIANT}_${tag}_seed${SEED}"
  if [[ -s "${out}/training_summary.json" ]]; then
    echo "Completed experiment exists; skipped: ${out}"
    return
  fi
  mkdir -p "${out}"
  [[ -s "${out}/latest.pth" ]] && resume+=(--resume "${out}/latest.pth")
  python -u -m bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.train \
    --variant "${VARIANT}" --parent_ckpt "${A1_CKPT}" \
    --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
    --output_dir "${out}" --optimizer_steps "${steps}" --eval_every "${eval_every}" \
    --save_every "${eval_every}" --val_max_batches "${val_max}" \
    --batch_size "${BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" --num_workers "${NUM_WORKERS:-8}" \
    --plugin_lr "${PLUGIN_LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --region_stride "${REGION_STRIDE:-8}" --device "${DEVICE:-cuda:0}" \
    --amp_dtype "${AMP_DTYPE:-bf16}" --seed "${SEED}" "${resume[@]}"
}

audit_one() {
  local target="$1" tag="${TRAIN_TAG:-probe5000}" checkpoint kind directory
  if [[ "${target}" == "c0_a1" ]]; then
    checkpoint="${A1_CKPT}"; kind="parent"; directory="${AUDIT_ROOT}/c0_a1_seed${SEED}"
  else
    checkpoint="${OUTPUT_ROOT}/${target}_${tag}_seed${SEED}/best_val_miou.pth"
    kind="experiment"; directory="${AUDIT_ROOT}/${target}_${tag}_seed${SEED}"
  fi
  test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 4; }
  mkdir -p "${directory}"
  for perturbation in none x4 y4 x8 y8 x16 y16; do
    [[ -s "${directory}/${perturbation}.json" && "${FORCE_AUDIT:-0}" != "1" ]] && continue
    python -u -m bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.audit \
      --variant "${target}" --checkpoint "${checkpoint}" --checkpoint_kind "${kind}" \
      --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
      --perturbation "${perturbation}" --output_json "${directory}/${perturbation}.json" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
      --region_stride "${REGION_STRIDE:-8}" --device "${DEVICE:-cuda:0}"
  done
}

case "${MODE}" in
  preflight) preflight ;;
  smoke|probe|confirm) preflight; train_one "${MODE}" ;;
  audit) preflight; audit_one "${VARIANT}" ;;
  audit_a1) preflight; audit_one c0_a1 ;;
  audit_all)
    preflight; audit_one c0_a1
    for value in c1_lca c2_slico c3_rgb_assoc c4_raw_sar_assoc c5_aligned_assoc c6_no_confidence c7_shuffled_sar c8_damage_hint c9_boundary; do audit_one "${value}"; done
    ;;
  summarize)
    python -m bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.summarize \
      --audit_root "${AUDIT_ROOT}" --output_json "${AUDIT_ROOT}/summary.json"
    ;;
  status)
    pgrep -af 'damageformer_a1_cross_modal_superpixel_v1.train' || true
    find "${OUTPUT_ROOT}" -maxdepth 2 -name training_summary.json -print 2>/dev/null || true
    ;;
  *) echo "mode must be preflight|smoke|probe|confirm|audit|audit_a1|audit_all|summarize|status" >&2; exit 2 ;;
esac

