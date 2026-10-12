#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
[[ "${MODE}" == "screen" ]] || {
  echo "H6-A2-DG ablation sequence supports only 'screen' mode" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
H6_A2_CKPT="${H6_A2_CKPT:-${OUTPUT_ROOT}/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/H6_A2_DG_ablation_${STAMP}.tsv"
printf "run\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

test -s "${H6_A2_CKPT}" || {
  echo "Missing locked H6-A2 validation baseline: ${H6_A2_CKPT}" >&2
  exit 1
}

note "shared H6-A2-DG preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/H6_A2_DG_preflight_${STAMP}.log"
if bash run/run_H6_A2_DG_5090.sh preflight >"${PREFLIGHT_LOG}" 2>&1; then
  note "shared preflight passed"
else
  note "preflight failed; see ${PREFLIGHT_LOG}"
  exit 1
fi

failures=0
for index in 0 1 2 3; do
  case "${index}" in
    0) label=control;  event_power=0.00; focus_probability=0.00 ;;
    1) label=focus;    event_power=0.00; focus_probability=0.35 ;;
    2) label=event;    event_power=0.50; focus_probability=0.00 ;;
    3) label=combined; event_power=0.50; focus_probability=0.35 ;;
  esac

  smoke_name="H6_A2_DG${index}_${label}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "DG${index}: smoke starts"
  if env \
      SEED="${SEED}" EXP_NAME="${smoke_name}" FORCE_RERUN="${FORCE_RERUN}" \
      EVENT_BALANCE_POWER="${event_power}" \
      FOCUS_PROBABILITY="${focus_probability}" \
      bash run/run_H6_A2_DG_5090.sh smoke >"${smoke_log}" 2>&1; then
    note "DG${index}: smoke passed"
    printf "DG%s\tsmoke\tpassed\t1\t%s\n" "${index}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "DG${index}: smoke failed; see ${smoke_log}"
    printf "DG%s\tsmoke\tfailed\t1\t%s\n" "${index}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  experiment="H6_A2_DG${index}_${label}${OPTIMIZER_STEPS}_seed${SEED}"
  completed=0
  for ((attempt=1; attempt<=MAX_RETRIES+1; attempt++)); do
    train_log="${LOG_ROOT}/${experiment}_attempt${attempt}.log"
    note "DG${index}: validation-only ${OPTIMIZER_STEPS}-step attempt ${attempt} starts"
    if env \
        SEED="${SEED}" EXP_NAME="${experiment}" FORCE_RERUN="${FORCE_RERUN}" \
        OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" EVAL_EVERY="${EVAL_EVERY}" \
        SAVE_EVERY="${OPTIMIZER_STEPS}" ENABLE_TEST_MONITOR=0 \
        EVENT_BALANCE_POWER="${event_power}" \
        FOCUS_PROBABILITY="${focus_probability}" \
        bash run/run_H6_A2_DG_5090.sh pilot >"${train_log}" 2>&1; then
      note "DG${index}: screening completed"
      printf "DG%s\tscreen\tcompleted\t%s\t%s\n" \
        "${index}" "${attempt}" "${train_log}" >> "${SUMMARY}"
      completed=1
      break
    fi
    note "DG${index}: attempt ${attempt} failed; see ${train_log}"
    printf "DG%s\tscreen\tfailed\t%s\t%s\n" \
      "${index}" "${attempt}" "${train_log}" >> "${SUMMARY}"
  done
  if [[ "${completed}" != "1" ]]; then
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

REPORT="${OUTPUT_ROOT}/H6_A2_DG_ablation${OPTIMIZER_STEPS}_seed${SEED}.json"
python -u script/standard_ML/summarize_H6_A2_DG_ablation.py \
  --output_root "${OUTPUT_ROOT}" \
  --baseline_checkpoint "${H6_A2_CKPT}" \
  --steps "${OPTIMIZER_STEPS}" \
  --seed "${SEED}" \
  --output_json "${REPORT}"

note "H6-A2-DG ablation finished; summary=${SUMMARY}; report=${REPORT}; failures=${failures}"
exit "${failures}"
