#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
D0_DIR="${D0_DIR:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/D0_original_damageformer_final50000_seed3407}"
A0_DIR="${A0_DIR:-/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a0_long35000_seed3407}"
A1_DIR="${A1_DIR:-/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407}"
LEGACY_RESULT_DIR="${LEGACY_RESULT_DIR:-/root/data/BRIGHT-eval-results/D0_A0_TEST878_LOCKED_V1}"
RESULT_DIR="${RESULT_DIR:-/root/data/BRIGHT-eval-results/D0_A0_A1_TEST878_LOCKED_V1}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "${BDA_ROOT}"
export PYTHONPATH="$(dirname "${BDA_ROOT}"):${BDA_ROOT}:${PYTHONPATH:-}"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    experiments/p3_damageformer_ablation_v1/evaluate.py \
    experiments/d0_a0_a1_test_locked_v1/compare_locked_test.py
  test -s "${TEST_LIST}"
  test -s "${D0_DIR}/best_val_miou.pth"
  test -s "${A0_DIR}/best_val_miou.pth"
  test -s "${A1_DIR}/best_val_miou.pth"
  test -s "${A1_DIR}/training_summary.json" || {
    echo "A1 is not a completed frozen experiment: ${A1_DIR}" >&2
    exit 4
  }
  echo "D0/A0/A1 locked Test878 preflight passed. No Test image was read."
}

evaluate_one() {
  local variant="$1"
  local checkpoint_kind="$2"
  local checkpoint="$3"
  local output="$4"
  if [[ -s "${output}" ]]; then
    echo "Preserving existing frozen result: ${output}"
    return
  fi
  local kind_args=()
  if [[ "${checkpoint_kind}" == "parent_damageformer" ]]; then
    kind_args+=(--checkpoint_kind parent_damageformer)
  fi
  "${PYTHON_BIN}" -u experiments/p3_damageformer_ablation_v1/evaluate.py \
    --variant "${variant}" "${kind_args[@]}" --checkpoint "${checkpoint}" \
    --dataset_path "${DATASET_ROOT}" --val_list "${TEST_LIST}" \
    --output_json "${output}" --allow_test_access \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
}

test_locked() {
  [[ "${ALLOW_TEST_ACCESS:-NO}" == "YES" ]] || {
    echo "Test is locked. Set ALLOW_TEST_ACCESS=YES only after freezing D0/A0/A1." >&2
    exit 3
  }
  preflight
  mkdir -p "${RESULT_DIR}"
  local d0_json="${RESULT_DIR}/d0_test878_locked.json"
  local a0_json="${RESULT_DIR}/a0_test878_locked.json"
  local a1_json="${RESULT_DIR}/a1_test878_locked.json"

  # Reuse the prior immutable D0/A0 outputs when present, avoiding repeat Test inference.
  if [[ ! -s "${d0_json}" && -s "${LEGACY_RESULT_DIR}/d0_test878_locked.json" ]]; then
    cp "${LEGACY_RESULT_DIR}/d0_test878_locked.json" "${d0_json}"
    echo "Reused frozen D0 result from the earlier locked comparison."
  fi
  if [[ ! -s "${a0_json}" && -s "${LEGACY_RESULT_DIR}/a0_test878_locked.json" ]]; then
    cp "${LEGACY_RESULT_DIR}/a0_test878_locked.json" "${a0_json}"
    echo "Reused frozen A0 result from the earlier locked comparison."
  fi

  evaluate_one a0 parent_damageformer "${D0_DIR}/best_val_miou.pth" "${d0_json}"
  evaluate_one a0 ablation "${A0_DIR}/best_val_miou.pth" "${a0_json}"
  evaluate_one a1 ablation "${A1_DIR}/best_val_miou.pth" "${a1_json}"

  "${PYTHON_BIN}" -u \
    experiments/d0_a0_a1_test_locked_v1/compare_locked_test.py \
    --d0 "${d0_json}" --a0 "${a0_json}" --a1 "${a1_json}" \
    --output "${RESULT_DIR}/d0_a0_a1_test878_locked_comparison.json"
}

case "${MODE}" in
  preflight) preflight ;;
  test_locked) test_locked ;;
  *) echo "mode must be preflight|test_locked" >&2; exit 2 ;;
esac
