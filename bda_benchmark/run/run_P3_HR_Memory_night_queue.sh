#!/usr/bin/env bash
set -uo pipefail

# Sequential overnight queue.  Every stage is isolated and Test is unavailable.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-7341}"
P3_CHECKPOINT="${P3_CHECKPOINT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/P3_HIERARCHICAL_RESIDUAL}"
LOG_ROOT="${LOG_ROOT:-/root/data/BRIGHT-train-logs/P3_HR_NIGHT_seed${SEED}}"
MEMORY_OUTPUT_ROOT="${MEMORY_OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed${SEED}}"
MEMORY_RESULT_ROOT="${MEMORY_RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2}"
D1_CACHE_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
LOCK_DIR="${OUTPUT_ROOT}/.night_queue_seed${SEED}.lock"
MANIFEST="${LOG_ROOT}/manifest.tsv"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Night queue lock already exists: ${LOCK_DIR}" >&2
  echo "Check for an existing queue before removing a stale lock." >&2
  exit 9
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT INT TERM

export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export SEED P3_CHECKPOINT OUTPUT_ROOT
export NUM_WORKERS="${NUM_WORKERS:-4}"

printf 'stage\tstatus\texit_code\tresult\tlog\n' > "${MANIFEST}"

record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "${MANIFEST}"
}

run_stage() {
  local stage="$1" result="$2" log="$3"
  shift 3
  if test -s "${result}"; then
    echo "[skip] ${stage}: existing ${result}"
    record "${stage}" "skipped_existing" 0 "${result}" "${log}"
    return 0
  fi
  echo "[start] ${stage}"
  "$@" > "${log}" 2>&1
  local code=$?
  if test "${code}" -eq 0 && test -s "${result}"; then
    echo "[done] ${stage}: ${result}"
    record "${stage}" "completed" 0 "${result}" "${log}"
  else
    echo "[failed] ${stage}: exit=${code}; see ${log}" >&2
    record "${stage}" "failed" "${code}" "${result}" "${log}"
  fi
  return "${code}"
}

PREFLIGHT_LOG="${LOG_ROOT}/00_preflight.log"
echo "[start] preflight"
bash run/run_P3_HierarchicalResidual.sh preflight > "${PREFLIGHT_LOG}" 2>&1
preflight_code=$?
if test "${preflight_code}" -ne 0; then
  record "preflight" "failed" "${preflight_code}" "-" "${PREFLIGHT_LOG}"
  echo "Preflight failed; queue stopped." >&2
  exit "${preflight_code}"
fi
record "preflight" "completed" 0 "-" "${PREFLIGHT_LOG}"

GUARD_RESULT="${OUTPUT_ROOT}/P3_HR0_exact_guard_seed${SEED}/training_summary.json"
run_stage "hr0_guard" "${GUARD_RESULT}" "${LOG_ROOT}/01_hr0_guard.log" \
  bash run/run_P3_HierarchicalResidual.sh guard
guard_code=$?
if test "${guard_code}" -ne 0; then
  echo "H0 failed; dependent model experiments were not run." >&2
  exit "${guard_code}"
fi
python - "${GUARD_RESULT}" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["decision"] == "ADMIT_TO_P3_HR1_500_STEP_SCREEN", row
assert row["exact_online_guard"] is True, row
assert row["test_accessed"] is False, row
PY

M1_JSON="${MEMORY_OUTPUT_ROOT}/m1_screen.json"
MEMORY_RESULT="${MEMORY_RESULT_ROOT}/m1r_selective_rescue_seed${SEED}.json"
if test -s "${M1_JSON}"; then
  run_stage "memory_m1r" "${MEMORY_RESULT}" "${LOG_ROOT}/02_memory_m1r.log" \
    env D1_CACHE_ROOT="${D1_CACHE_ROOT}" \
      OUTPUT_ROOT="${MEMORY_OUTPUT_ROOT}" RESULT_ROOT="${MEMORY_RESULT_ROOT}" \
      MEMORY_TOPK=4 RETRIEVAL_TEMPERATURE=0.10 MINIMUM_MEMORY_PURITY=0.0 \
      bash experiments/spi_cross_event_metric_memory_v2/run.sh rescue
else
  echo "[skip] memory_m1r: missing ${M1_JSON}"
  record "memory_m1r" "skipped_missing_m1" 0 "${MEMORY_RESULT}" "${LOG_ROOT}/02_memory_m1r.log"
fi

for variant in hr1 hr1o hr1s hr2 hr3o; do
  upper="$(printf '%s' "${variant}" | tr '[:lower:]' '[:upper:]')"
  result="${OUTPUT_ROOT}/P3_${upper}_screen500_seed${SEED}/training_summary.json"
  run_stage "${variant}" "${result}" "${LOG_ROOT}/${variant}.log" \
    bash run/run_P3_HierarchicalResidual.sh "${variant}"
done

python -u script/standard_ML/summarize_P3_HR_night.py \
  --output-root "${OUTPUT_ROOT}" --memory-result "${MEMORY_RESULT}" \
  --seed "${SEED}" --output "${LOG_ROOT}/summary.json" \
  > "${LOG_ROOT}/summary.txt" 2>&1
summary_code=$?
record "summary" "$([ "${summary_code}" -eq 0 ] && echo completed || echo failed)" \
  "${summary_code}" "${LOG_ROOT}/summary.json" "${LOG_ROOT}/summary.txt"

echo "Night queue complete."
echo "Manifest: ${MANIFEST}"
echo "Summary : ${LOG_ROOT}/summary.txt"
