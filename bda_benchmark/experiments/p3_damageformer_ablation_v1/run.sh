#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-a0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${VARIANT}" in a0|a1|a2|a3) ;; *) echo "variant must be a0|a1|a2|a3" >&2; exit 2 ;; esac

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-}"
PARENT_KIND="${PARENT_KIND:-damageformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/P3_DAMAGEFORMER_ABLATION_V1}"
SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    experiments/p3_damageformer_ablation_v1/core.py \
    experiments/p3_damageformer_ablation_v1/inspect_parent.py \
    experiments/p3_damageformer_ablation_v1/train.py \
    experiments/p3_damageformer_ablation_v1/evaluate.py \
    experiments/p3_damageformer_ablation_v1/summarize.py
  python -m unittest experiments/p3_damageformer_ablation_v1/test_contract.py -v
  test -s "${TRAIN_LIST}"
  test -s "${VAL_LIST}"
  echo "P3 DamageFormer ablation preflight passed. Test split will not be accessed."
}

stage_args() {
  case "$1" in
    smoke) echo "smoke20 20 10 2 none" ;;
    probe) echo "probe5000 5000 500 0 none" ;;
    confirm) echo "confirm20000 20000 1000 0 probe5000" ;;
    long) echo "long35000 35000 1000 0 confirm20000,probe5000" ;;
    full) echo "full50000 50000 1000 0 long35000,confirm20000,probe5000" ;;
    *) echo "mode must be preflight|smoke|probe|confirm|long|full|eval|summarize" >&2; exit 2 ;;
  esac
}

train_one() {
  : "${PARENT_CKPT:?Set PARENT_CKPT to one exact completed parent checkpoint}"
  test -s "${PARENT_CKPT}"
  read -r tag steps eval_every val_max previous_tag <<<"$(stage_args "${MODE}")"
  out="${OUTPUT_ROOT}/${VARIANT}_${tag}_seed${SEED}"
  if [[ -s "${out}/training_summary.json" ]]; then
    echo "Completed experiment exists; skipped: ${out}"
    return
  fi
  mkdir -p "${out}"
  resume_args=()
  if [[ -s "${out}/latest_resume.pth" ]]; then
    resume_args+=(--resume "${out}/latest_resume.pth")
  elif [[ "${previous_tag}" != "none" ]]; then
    for candidate_tag in ${previous_tag//,/ }; do
      previous_dir="${OUTPUT_ROOT}/${VARIANT}_${candidate_tag}_seed${SEED}"
      if [[ -s "${previous_dir}/latest_resume.pth" ]]; then
        if [[ -s "${previous_dir}/best_val_miou.pth" && ! -s "${out}/best_val_miou.pth" ]]; then
          cp "${previous_dir}/best_val_miou.pth" "${out}/best_val_miou.pth"
        fi
        resume_args+=(--resume "${previous_dir}/latest_resume.pth")
        echo "Continuing ${VARIANT} from ${previous_dir}/latest_resume.pth"
        break
      fi
    done
  fi
  python -u experiments/p3_damageformer_ablation_v1/train.py \
    --variant "${VARIANT}" --dataset_path "${DATASET_PATH}" \
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
    --parent_ckpt "${PARENT_CKPT}" --parent_kind "${PARENT_KIND}" --output_dir "${out}" \
    --optimizer_steps "${steps}" --eval_every "${eval_every}" \
    --val_max_batches "${val_max}" --batch_size "${BATCH_SIZE:-4}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --core_lr "${CORE_LR:-1e-5}" --bag_lr "${BAG_LR:-1e-4}" \
    --amp_dtype "${AMP_DTYPE:-bf16}" --seed "${SEED}" "${resume_args[@]}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke|probe|confirm|long|full) train_one ;;
  eval)
    TAG="${TRAIN_TAG:-probe5000}"
    CKPT="${OUTPUT_ROOT}/${VARIANT}_${TAG}_seed${SEED}/best_val_miou.pth"
    test -s "${CKPT}"
    mkdir -p "${RESULT_ROOT}/${VARIANT}_${TAG}_seed${SEED}"
    python -u experiments/p3_damageformer_ablation_v1/evaluate.py \
      --variant "${VARIANT}" --checkpoint "${CKPT}" \
      --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
      --output_json "${RESULT_ROOT}/${VARIANT}_${TAG}_seed${SEED}/val.json" \
      --num_workers "${NUM_WORKERS:-8}"
    ;;
  summarize)
    mkdir -p "${RESULT_ROOT}"
    python experiments/p3_damageformer_ablation_v1/summarize.py \
      --output_root "${OUTPUT_ROOT}" --tag "${TRAIN_TAG:-probe5000}" --seed "${SEED}"
    ;;
  *) stage_args "${MODE}" >/dev/null ;;
esac
