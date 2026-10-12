#!/usr/bin/env bash
set -uo pipefail

# Fault-tolerant A0->A4 controller. Each variant receives an independent smoke
# test, experiment directory and log. A failed main run is retried once and
# resumes automatically from latest_resume.pth.

MODE="${1:-pilot}"
case "${MODE}" in
  pilot|full) ;;
  *)
    echo "Usage: bash run/run_S8_P4_NSMICB_sequence.sh pilot|full" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_P4_NSMICB}"
VARIANTS="${VARIANTS:-a0 a1 a2 a3 a4}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
SEED="${SEED:-3407}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
printf "variant\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_shared() {
  local stage="$1"
  local logfile="${LOG_ROOT}/shared_${stage}_${STAMP}.log"
  log "shared ${stage} starts"
  if env OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash run/run_S8_P4_NSMICB_5090.sh "${stage}" \
      > "${logfile}" 2>&1; then
    log "shared ${stage} passed"
    return 0
  fi
  log "shared ${stage} failed; see ${logfile}"
  return 1
}

if ! run_shared preflight; then
  exit 10
fi
if ! run_shared stats; then
  exit 11
fi
if ! run_shared guard; then
  exit 12
fi

overall_failure=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  smoke_name="S8_P4_${upper}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  log "${variant}: smoke starts"
  if env \
      VARIANT="${variant}" \
      EXP_NAME="${smoke_name}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SEED="${SEED}" \
      bash run/run_S8_P4_NSMICB_5090.sh smoke \
      > "${smoke_log}" 2>&1; then
    log "${variant}: smoke passed"
    printf "%s\tsmoke\tpassed\t1\t%s\n" \
      "${variant}" "${smoke_log}" >> "${SUMMARY}"
  else
    log "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" \
      "${variant}" "${smoke_log}" >> "${SUMMARY}"
    overall_failure=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      break
    fi
    continue
  fi

  if [[ "${MODE}" == "pilot" ]]; then
    mode_tag="pilot3000"
  else
    mode_tag="full50000"
  fi
  exp_name="S8_P4_${upper}_${mode_tag}_seed${SEED}"
  main_log="${LOG_ROOT}/${exp_name}.log"
  attempt=0
  completed=0
  while (( attempt <= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    log "${variant}: ${MODE} attempt ${attempt} starts"
    if env \
        VARIANT="${variant}" \
        EXP_NAME="${exp_name}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        SEED="${SEED}" \
        bash run/run_S8_P4_NSMICB_5090.sh "${MODE}" \
        >> "${main_log}" 2>&1; then
      completed=1
      log "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\t%s\n" \
        "${variant}" "${MODE}" "${attempt}" "${main_log}" >> "${SUMMARY}"
      break
    fi
    log "${variant}: ${MODE} attempt ${attempt} failed"
  done

  if [[ "${completed}" != "1" ]]; then
    overall_failure=1
    printf "%s\t%s\tfailed\t%s\t%s\n" \
      "${variant}" "${MODE}" "${attempt}" "${main_log}" >> "${SUMMARY}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      break
    fi
  fi
done

if [[ "${MODE}" == "pilot" ]]; then
  summary_tag="pilot3000"
else
  summary_tag="full50000"
fi
python script/standard_ML/summarize_S8_P4_ablation.py \
  --output_root "${OUTPUT_ROOT}" \
  --mode_tag "${summary_tag}" \
  --seed "${SEED}" \
  --variants "${VARIANTS}" \
  --output_json "${LOG_ROOT}/ablation_${summary_tag}_seed${SEED}.json" \
  --output_tsv "${LOG_ROOT}/ablation_${summary_tag}_seed${SEED}.tsv" \
  >> "${LOG_ROOT}/ablation_${summary_tag}_seed${SEED}.log" 2>&1 || true

log "sequence finished; summary=${SUMMARY}"
exit "${overall_failure}"
