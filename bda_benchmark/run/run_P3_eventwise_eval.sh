#!/usr/bin/env bash
set -euo pipefail

# One-pass, 14-event official evaluation for the locked P3 checkpoint.
# Validation is the default.  Test requires the explicit one-time ALLOW_TEST=1 lock.

MODE="${1:-val}"
case "${MODE}" in
  preflight|val|test) ;;
  *) echo "Usage: bash run/run_P3_eventwise_eval.sh preflight|val|test" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/root/shared-nvme/BRIGHT-master/bda_benchmark}"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/P3_EVENTWISE}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
BASELINE_EVENT_CSV="${BASELINE_EVENT_CSV:-}"

EVALUATOR="${REPO_ROOT}/full_evaluate_unified.py"
SUMMARIZER="${PROJECT_ROOT}/script/standard_ML/summarize_P3_eventwise_eval.py"
TEST_MODULE="tests.test_p3_eventwise_eval"

for required in "${P3_CKPT}" "${EVALUATOR}" "${SUMMARIZER}" "${DATASET_ROOT}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

actual_sha256="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_P3_SHA256}" ]]; then
  echo "Locked P3 checkpoint SHA256 mismatch." >&2
  echo "Expected: ${EXPECTED_P3_SHA256}" >&2
  echo "Observed: ${actual_sha256}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m py_compile "${SUMMARIZER}"
python -m unittest -v "${TEST_MODULE}"
if [[ "${MODE}" == "preflight" ]]; then
  echo "P3 event-wise evaluation preflight passed. Test was not accessed."
  exit 0
fi

if [[ "${MODE}" == "test" && "${ALLOW_TEST:-0}" != "1" ]]; then
  echo "Test is locked. Re-run with ALLOW_TEST=1 only for the single declared final evaluation." >&2
  exit 3
fi

if [[ "${MODE}" == "val" ]]; then
  LIST_PATH="${VAL_LIST}"
else
  LIST_PATH="${TEST_LIST}"
fi
[[ -s "${LIST_PATH}" ]] || { echo "Split list is missing or empty: ${LIST_PATH}" >&2; exit 1; }

OUT_DIR="${OUTPUT_ROOT}/${MODE}/S8_MICB_DSMIL_P3"
mkdir -p "${OUT_DIR}"
printf '%s  %s\n' "${actual_sha256}" "${P3_CKPT}" > "${OUT_DIR}/checkpoint_sha256.txt"

if [[ "${FORCE_EVAL:-0}" == "1" || ! -s "${OUT_DIR}/summary_official.json" || ! -s "${OUT_DIR}/event_confusion_4class.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python -u "${EVALUATOR}" \
    --eval_protocol official \
    --model_type DamageFormer_S8_MICB_DSMIL \
    --model_path "${P3_CKPT}" \
    --strict_checkpoint \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${LIST_PATH}" \
    --image_size 1024 \
    --batch_size "${EVAL_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --save_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/eval.log"
else
  echo "Official inference cache found; reusing ${OUT_DIR}. Set FORCE_EVAL=1 to recompute."
fi

summary_args=(
  --eval_dir "${OUT_DIR}"
  --split "${MODE}"
  --checkpoint_sha256 "${actual_sha256}"
)
if [[ -n "${BASELINE_EVENT_CSV}" ]]; then
  summary_args+=(--baseline_event_csv "${BASELINE_EVENT_CSV}")
fi
if [[ "${MODE}" == "val" ]]; then
  summary_args+=(
    --expected_event_mean "${EXPECTED_VAL_EVENT_MEAN:-52.73}"
    --expected_event_worst "${EXPECTED_VAL_EVENT_WORST:-40.73}"
    --expected_tolerance "${EXPECTED_EVENT_TOLERANCE:-0.02}"
  )
fi
python -u "${SUMMARIZER}" "${summary_args[@]}" | tee "${OUT_DIR}/eventwise_summary.log"

echo "Readable table: ${OUT_DIR}/eventwise_summary.md"
echo "Machine table : ${OUT_DIR}/eventwise_summary.tsv"
echo "Full JSON     : ${OUT_DIR}/eventwise_summary.json"
