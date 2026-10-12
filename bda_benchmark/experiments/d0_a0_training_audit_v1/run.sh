#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-curves}"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${BDA_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
SAR_STATS="${SAR_STATS:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}"
D0_DIR="${D0_DIR:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/D0_original_damageformer_final50000_seed3407}"
A0_DIR="${A0_DIR:-/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a0_long35000_seed3407}"
D0_LOG="${D0_LOG:-/root/data/BRIGHT-train-logs/DAMAGEFORMER_PARENT_V1/train_final50000.log}"
A0_LOG="${A0_LOG:-/root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1/a0_long_seed3407.log}"
RESULT_DIR="${RESULT_DIR:-/root/data/BRIGHT-eval-results/D0_A0_TRAINING_AUDIT_V1}"
TEST_RESULT_DIR="${TEST_RESULT_DIR:-/root/data/BRIGHT-eval-results/D0_A0_TEST878_LOCKED_V1}"

mkdir -p "${RESULT_DIR}"
cd "${BDA_ROOT}"
export PYTHONPATH="$(dirname "${BDA_ROOT}"):${BDA_ROOT}:${PYTHONPATH:-}"

curves() {
  local extra=()
  if [[ -n "${D0_EVAL_JSON:-}" ]]; then
    extra+=(--d0-eval-json "${D0_EVAL_JSON}")
  fi
  if [[ -n "${A0_EVAL_JSON:-}" ]]; then
    extra+=(--a0-eval-json "${A0_EVAL_JSON}")
  fi
  "${PYTHON_BIN}" experiments/d0_a0_training_audit_v1/summarize_training.py \
    --d0-log "${D0_LOG}" --d0-dir "${D0_DIR}" \
    --a0-log "${A0_LOG}" --a0-dir "${A0_DIR}" \
    --output-dir "${RESULT_DIR}" "${extra[@]}"
}

evaluate() {
  test -s "${D0_DIR}/best_val_miou.pth"
  test -s "${A0_DIR}/best_val_miou.pth"
  "${PYTHON_BIN}" -u experiments/salt_cde_validation_v1/audit_model.py \
    --checkpoint "${D0_DIR}/best_val_miou.pth" \
    --dataset_path "${DATASET_ROOT}" --data_list_path "${VAL_LIST}" \
    --sar_stats_path "${SAR_STATS}" --perturbation none \
    --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}" --output_json "${RESULT_DIR}/d0_native_val450.json"
  "${PYTHON_BIN}" -u experiments/p3_damageformer_ablation_v1/evaluate.py \
    --variant a0 --checkpoint_kind parent_damageformer \
    --checkpoint "${D0_DIR}/best_val_miou.pth" \
    --dataset_path "${DATASET_ROOT}" --val_list "${VAL_LIST}" \
    --output_json "${RESULT_DIR}/d0_as_a0_step0_val450.json" \
    --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}"
  "${PYTHON_BIN}" -u experiments/p3_damageformer_ablation_v1/evaluate.py \
    --variant a0 --checkpoint "${A0_DIR}/best_val_miou.pth" \
    --dataset_path "${DATASET_ROOT}" --val_list "${VAL_LIST}" \
    --output_json "${RESULT_DIR}/a0_val450.json" \
    --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
    --device "${DEVICE:-cuda:0}"
  D0_EVAL_JSON="${RESULT_DIR}/d0_as_a0_step0_val450.json" \
  A0_EVAL_JSON="${RESULT_DIR}/a0_val450.json" curves
}

test_locked() {
  [[ "${ALLOW_TEST_ACCESS:-NO}" == "YES" ]] || {
    echo "Test is locked. Set ALLOW_TEST_ACCESS=YES only after freezing D0 and A0." >&2
    exit 3
  }
  test -s "${D0_DIR}/best_val_miou.pth"
  test -s "${A0_DIR}/best_val_miou.pth"
  test -s "${TEST_LIST}"
  mkdir -p "${TEST_RESULT_DIR}"
  local d0_json="${TEST_RESULT_DIR}/d0_test878_locked.json"
  local a0_json="${TEST_RESULT_DIR}/a0_test878_locked.json"
  local comparison="${TEST_RESULT_DIR}/d0_vs_a0_test878_locked.json"
  if [[ ! -s "${d0_json}" ]]; then
    "${PYTHON_BIN}" -u experiments/p3_damageformer_ablation_v1/evaluate.py \
      --variant a0 --checkpoint_kind parent_damageformer \
      --checkpoint "${D0_DIR}/best_val_miou.pth" \
      --dataset_path "${DATASET_ROOT}" --val_list "${TEST_LIST}" \
      --output_json "${d0_json}" --allow_test_access \
      --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
      --device "${DEVICE:-cuda:0}"
  else
    echo "Preserving existing locked D0 Test result: ${d0_json}"
  fi
  if [[ ! -s "${a0_json}" ]]; then
    "${PYTHON_BIN}" -u experiments/p3_damageformer_ablation_v1/evaluate.py \
      --variant a0 --checkpoint "${A0_DIR}/best_val_miou.pth" \
      --dataset_path "${DATASET_ROOT}" --val_list "${TEST_LIST}" \
      --output_json "${a0_json}" --allow_test_access \
      --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
      --device "${DEVICE:-cuda:0}"
  else
    echo "Preserving existing locked A0 Test result: ${a0_json}"
  fi
  "${PYTHON_BIN}" -u experiments/d0_a0_training_audit_v1/compare_locked_test.py \
    --d0 "${d0_json}" --a0 "${a0_json}" --output "${comparison}"
  echo "Test accessed: YES. Do not tune D0 or A0 from this result."
}

case "${MODE}" in
  curves) curves ;;
  evaluate) evaluate ;;
  test_locked) test_locked ;;
  *) echo "mode must be curves|evaluate|test_locked" >&2; exit 2 ;;
esac
