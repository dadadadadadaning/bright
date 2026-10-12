#!/usr/bin/env bash
set -euo pipefail

# Fair, strict comparison of:
#   1) S8 MICB-DSMIL P3 (formal experiment: S8_MICB_DSMIL_P3)
#   2) S8 IA-MLA
#
# Usage:
#   bash run/run_eval_S8_pair.sh preflight
#   bash run/run_eval_S8_pair.sh smoke
#   bash run/run_eval_S8_pair.sh test
#   bash run/run_eval_S8_pair.sh val
#   bash run/run_eval_S8_pair.sh both

MODE="${1:-test}"
case "${MODE}" in
  preflight|smoke|val|test|both) ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|val|test|both" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master/bda_benchmark}"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
SMOKE_LIST="${SMOKE_LIST:-}"

P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
IA_CKPT="${IA_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8/BRIGHT/S8_IA_MLA_EBS16_50k/best_valid_miou.pth}"
S8_PAIR_OUTPUT_ROOT="${S8_PAIR_OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/S8_pair}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

EVAL_SCRIPT="${REPO_ROOT}/full_evaluate_unified.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/script/standard_ML/summarize_S8_pair_eval.py"
INSPECT_SCRIPT="${PROJECT_ROOT}/script/standard_ML/inspect_S8_pair_checkpoints.py"

for required in \
  "${P3_CKPT}" \
  "${IA_CKPT}" \
  "${EVAL_SCRIPT}" \
  "${SUMMARY_SCRIPT}" \
  "${INSPECT_SCRIPT}" \
  "${DATASET_ROOT}"
do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 1
  fi
done

actual_p3_sha256="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
if [[ "${actual_p3_sha256}" != "${EXPECTED_P3_SHA256}" ]]; then
  echo "P3 checkpoint SHA256 mismatch." >&2
  echo "Expected: ${EXPECTED_P3_SHA256}" >&2
  echo "Observed: ${actual_p3_sha256}" >&2
  exit 1
fi

mkdir -p "${S8_PAIR_OUTPUT_ROOT}"
sha256sum "${P3_CKPT}" "${IA_CKPT}" | tee "${S8_PAIR_OUTPUT_ROOT}/checkpoint_sha256.txt"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python -m py_compile "${EVAL_SCRIPT}" "${SUMMARY_SCRIPT}" "${INSPECT_SCRIPT}"
python -u "${INSPECT_SCRIPT}" \
  --p3_checkpoint "${P3_CKPT}" \
  --ia_checkpoint "${IA_CKPT}" \
  --output_json "${S8_PAIR_OUTPUT_ROOT}/checkpoint_identity.json"
if [[ "${MODE}" == "preflight" ]]; then
  echo "Preflight passed: files, SHA256, syntax, and checkpoint identities are valid."
  exit 0
fi

evaluate_one() {
  local split="$1"
  local list_path="$2"
  local model_name="$3"
  local model_type="$4"
  local checkpoint="$5"
  local output_dir="${S8_PAIR_OUTPUT_ROOT}/${split}/${model_name}"

  if [[ ! -s "${list_path}" ]]; then
    echo "Data list is missing or empty: ${list_path}" >&2
    exit 1
  fi
  mkdir -p "${output_dir}"

  echo "================================================================"
  echo "Split       : ${split}"
  echo "Model       : ${model_name}"
  echo "Model type  : ${model_type}"
  echo "Checkpoint  : ${checkpoint}"
  echo "Output      : ${output_dir}"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python -u "${EVAL_SCRIPT}" \
    --eval_protocol official \
    --model_type "${model_type}" \
    --model_path "${checkpoint}" \
    --strict_checkpoint \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${list_path}" \
    --image_size 1024 \
    --batch_size "${EVAL_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --save_dir "${output_dir}" \
    2>&1 | tee "${output_dir}/eval.log"
}

evaluate_split() {
  local split="$1"
  local list_path="$2"

  evaluate_one \
    "${split}" \
    "${list_path}" \
    "S8_MICB_DSMIL_P3" \
    "DamageFormer_S8_MICB_DSMIL" \
    "${P3_CKPT}"

  evaluate_one \
    "${split}" \
    "${list_path}" \
    "S8_IA_MLA" \
    "DamageFormer_S8_IA_MLA" \
    "${IA_CKPT}"

  python -u "${SUMMARY_SCRIPT}" \
    --p3_json "${S8_PAIR_OUTPUT_ROOT}/${split}/S8_MICB_DSMIL_P3/summary_official.json" \
    --ia_json "${S8_PAIR_OUTPUT_ROOT}/${split}/S8_IA_MLA/summary_official.json" \
    --output_json "${S8_PAIR_OUTPUT_ROOT}/${split}/comparison.json" \
    --output_tsv "${S8_PAIR_OUTPUT_ROOT}/${split}/comparison.tsv" \
    2>&1 | tee "${S8_PAIR_OUTPUT_ROOT}/${split}/comparison.log"
}

if [[ "${MODE}" == "smoke" ]]; then
  if [[ -z "${SMOKE_LIST}" ]]; then
    SMOKE_LIST="${S8_PAIR_OUTPUT_ROOT}/smoke20_val_list.txt"
    head -n 20 "${VAL_LIST}" > "${SMOKE_LIST}"
  fi
  evaluate_split "smoke20_val" "${SMOKE_LIST}"
fi
if [[ "${MODE}" == "val" || "${MODE}" == "both" ]]; then
  evaluate_split "val" "${VAL_LIST}"
fi
if [[ "${MODE}" == "test" || "${MODE}" == "both" ]]; then
  evaluate_split "test" "${TEST_LIST}"
fi

echo "All requested evaluations completed."
echo "Results: ${S8_PAIR_OUTPUT_ROOT}"
