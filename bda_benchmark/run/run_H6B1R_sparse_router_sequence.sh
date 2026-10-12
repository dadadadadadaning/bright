#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
[[ "${MODE}" == pilot ]] || {
  echo "H6-B1R development supports only pilot mode" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEEDS="${SEEDS:-3407 2027 7777}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
MAX_RETRIES="${MAX_RETRIES:-1}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/H6B1R_sequence_${STAMP}.tsv"
printf "seed\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
run_logged() { local log="$1"; shift; "$@" > "${log}" 2>&1; }

note "shared H6-B1R preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/H6B1R_preflight_${STAMP}.log"
run_logged "${PREFLIGHT_LOG}" bash run/run_H6_SDCR_5090.sh preflight || {
  note "preflight failed; see ${PREFLIGHT_LOG}"
  exit 1
}
note "shared preflight passed"

note "P3 validation-only safety guard starts (test disabled)"
GUARD_LOG="${LOG_ROOT}/H6B1R_B0_val_guard_${STAMP}.log"
run_logged "${GUARD_LOG}" env \
  ENABLE_TEST_MONITOR=0 \
  EXP_NAME="H6_B0_B1R_val_guard_${STAMP}" \
  bash run/run_H6_SDCR_5090.sh safe_guard || {
  note "validation guard failed; see ${GUARD_LOG}"
  exit 1
}
note "validation guard passed"

failures=0
for seed in ${SEEDS}; do
  smoke_name="H6_B1R_smoke20_seed${seed}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "seed ${seed}: smoke starts (test disabled)"
  if run_logged "${smoke_log}" env \
      ENABLE_TEST_MONITOR=0 VARIANT=b1r SEED="${seed}" \
      EXP_NAME="${smoke_name}" NEW_LR=5e-5 WARMUP_STEPS=20 \
      bash run/run_H6_SDCR_5090.sh smoke; then
    note "seed ${seed}: smoke passed"
    printf "%s\tsmoke\tpassed\t1\t%s\n" "${seed}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "seed ${seed}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" "${seed}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1))
    continue
  fi

  experiment="H6_B1R_pilot${OPTIMIZER_STEPS}_seed${seed}"
  completed=0
  for ((attempt=1; attempt<=MAX_RETRIES+1; attempt++)); do
    log="${LOG_ROOT}/${experiment}_attempt${attempt}.log"
    note "seed ${seed}: validation-only pilot attempt ${attempt} starts"
    if run_logged "${log}" env \
        ENABLE_TEST_MONITOR=0 VARIANT=b1r SEED="${seed}" \
        EXP_NAME="${experiment}" OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" \
        EVAL_EVERY=250 SAVE_EVERY=500 WARMUP_STEPS="${OPTIMIZER_STEPS}" \
        NEW_LR=5e-5 FORCE_RERUN=0 \
        bash run/run_H6_SDCR_5090.sh pilot; then
      note "seed ${seed}: pilot completed"
      printf "%s\tpilot\tcompleted\t%s\t%s\n" \
        "${seed}" "${attempt}" "${log}" >> "${SUMMARY}"
      completed=1
      break
    fi
    note "seed ${seed}: attempt ${attempt} failed; see ${log}"
    printf "%s\tpilot\tfailed\t%s\t%s\n" \
      "${seed}" "${attempt}" "${log}" >> "${SUMMARY}"
  done
  [[ "${completed}" == 1 ]] || failures=$((failures + 1))
done

python -u script/standard_ML/summarize_H6B1R_seeds.py \
  --output_root "${OUTPUT_ROOT}" \
  --steps "${OPTIMIZER_STEPS}" \
  --seeds ${SEEDS} \
  --output_json "${OUTPUT_ROOT}/H6_B1R_validation_only_summary.json"

note "H6-B1R sequence finished; summary=${SUMMARY}; failures=${failures}"
exit "${failures}"
