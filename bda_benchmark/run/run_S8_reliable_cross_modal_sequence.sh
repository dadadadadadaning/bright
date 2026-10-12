#!/usr/bin/env bash
set -uo pipefail

# Fault-tolerant sequential controller for S8-R0...R5.
#
# Properties:
#   * one shared preflight before any GPU training;
#   * one full-validation R0 baseline guard with zero optimizer steps;
#   * a short smoke gate before every variant;
#   * isolated experiment directories and logs;
#   * automatic resume from latest_resume.pth;
#   * completed variants are skipped when this controller is restarted;
#   * failures are recorded and, by default, do not erase/stop other variants.
#
# Usage:
#   nohup bash run/run_S8_reliable_cross_modal_sequence.sh pilot \
#     > /root/shared-nvme/BRIGHT-train-logs/S8_ReliableCrossModal/controller.log 2>&1 &
#
# Optional:
#   VARIANTS="r0 r1 r2" MAX_RETRIES=1 CONTINUE_ON_FAILURE=1 \
#     bash run/run_S8_reliable_cross_modal_sequence.sh pilot

MODE="${1:-pilot}"
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "MODE must be pilot or full" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/run/run_S8_reliable_cross_modal_5090.sh"

VARIANTS="${VARIANTS:-r0 r1 r2 r3 r4 r5}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
BASE_CKPT="${BASE_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/S8_ReliableCrossModal}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_ReliableCrossModal}"
EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU:-68.6493}"
BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE:-0.5}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${RUN_ID}.tsv"
PREFLIGHT_LOG="${LOG_ROOT}/preflight_${RUN_ID}.log"
printf "variant\tphase\tstatus\texit_code\tartifact\tlog\n" > "${SUMMARY}"

stamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

record() {
  local variant="$1"
  local phase="$2"
  local status="$3"
  local exit_code="$4"
  local artifact="$5"
  local log_file="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" "${phase}" "${status}" "${exit_code}" "${artifact}" "${log_file}" \
    >> "${SUMMARY}"
}

echo "[$(stamp)] S8 sequence starts: mode=${MODE}; variants=${VARIANTS}"
echo "[$(stamp)] summary=${SUMMARY}"
echo "[$(stamp)] running shared preflight"
if env BASE_CKPT="${BASE_CKPT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash "${LAUNCHER}" preflight > "${PREFLIGHT_LOG}" 2>&1; then
  echo "[$(stamp)] preflight passed"
else
  code=$?
  echo "[$(stamp)] preflight FAILED (exit=${code}); see ${PREFLIGHT_LOG}" >&2
  record "all" "preflight" "failed" "${code}" "" "${PREFLIGHT_LOG}"
  exit "${code}"
fi

BASELINE_NAME="S8_R0_full_validation_guard"
BASELINE_DIR="${OUTPUT_ROOT}/${BASELINE_NAME}"
BASELINE_LOG="${LOG_ROOT}/${BASELINE_NAME}.log"
echo "[$(stamp)] running R0 full-validation baseline guard"
if [[ -s "${BASELINE_DIR}/final.pth" ]]; then
  echo "[$(stamp)] baseline guard already passed; reused"
  record "r0" "baseline_guard" "skipped_complete" "0" \
    "${BASELINE_DIR}/final.pth" "${BASELINE_LOG}"
elif env \
  BASE_CKPT="${BASE_CKPT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU}" \
  BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE}" \
  VARIANT="r0" \
  EXP_NAME="${BASELINE_NAME}" \
  bash "${LAUNCHER}" baseline > "${BASELINE_LOG}" 2>&1; then
  echo "[$(stamp)] baseline guard passed"
  record "r0" "baseline_guard" "passed" "0" \
    "${BASELINE_DIR}/final.pth" "${BASELINE_LOG}"
else
  code=$?
  echo "[$(stamp)] baseline guard FAILED (exit=${code}); ${BASELINE_LOG}" >&2
  record "r0" "baseline_guard" "failed" "${code}" "" "${BASELINE_LOG}"
  echo "[$(stamp)] all variants share this base; sequence is stopped safely" >&2
  exit "${code}"
fi

for variant in ${VARIANTS}; do
  upper="${variant^^}"
  smoke_name="S8_${upper}_smoke_sequence"
  smoke_dir="${OUTPUT_ROOT}/${smoke_name}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"

  echo "[$(stamp)] ${variant}: smoke check"
  if [[ -s "${smoke_dir}/final.pth" ]]; then
    echo "[$(stamp)] ${variant}: existing successful smoke is reused"
    record "${variant}" "smoke" "skipped_complete" "0" \
      "${smoke_dir}/final.pth" "${smoke_log}"
  else
    if env \
      BASE_CKPT="${BASE_CKPT}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU}" \
      BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE}" \
      VARIANT="${variant}" \
      EXP_NAME="${smoke_name}" \
      bash "${LAUNCHER}" smoke > "${smoke_log}" 2>&1; then
      echo "[$(stamp)] ${variant}: smoke passed"
      record "${variant}" "smoke" "passed" "0" \
        "${smoke_dir}/final.pth" "${smoke_log}"
    else
      code=$?
      echo "[$(stamp)] ${variant}: smoke FAILED (exit=${code}); ${smoke_log}" >&2
      record "${variant}" "smoke" "failed" "${code}" "" "${smoke_log}"
      if [[ "${CONTINUE_ON_FAILURE}" == "1" ]]; then
        continue
      fi
      exit "${code}"
    fi
  fi

  exp_name="S8_${upper}_${MODE}_sequence"
  exp_dir="${OUTPUT_ROOT}/${exp_name}"
  final_ckpt="${exp_dir}/final.pth"
  latest_ckpt="${exp_dir}/latest_resume.pth"

  if [[ -s "${final_ckpt}" ]]; then
    echo "[$(stamp)] ${variant}: ${MODE} already complete; skipped"
    record "${variant}" "${MODE}" "skipped_complete" "0" \
      "${final_ckpt}" "${LOG_ROOT}/${exp_name}.log"
    continue
  fi

  success=0
  attempt=0
  while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
    attempt=$((attempt + 1))
    attempt_log="${LOG_ROOT}/${exp_name}_attempt${attempt}.log"
    resume_args=()
    if [[ -s "${latest_ckpt}" ]]; then
      resume_args+=(RESUME_CKPT="${latest_ckpt}")
      echo "[$(stamp)] ${variant}: attempt ${attempt} resumes ${latest_ckpt}"
    else
      echo "[$(stamp)] ${variant}: attempt ${attempt} starts from shared base"
    fi

    if env \
      BASE_CKPT="${BASE_CKPT}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      EXPECTED_BASE_VAL_MIOU="${EXPECTED_BASE_VAL_MIOU}" \
      BASE_MIOU_TOLERANCE="${BASE_MIOU_TOLERANCE}" \
      VARIANT="${variant}" \
      EXP_NAME="${exp_name}" \
      "${resume_args[@]}" \
      bash "${LAUNCHER}" "${MODE}" > "${attempt_log}" 2>&1; then
      success=1
      echo "[$(stamp)] ${variant}: ${MODE} completed"
      record "${variant}" "${MODE}" "passed" "0" \
        "${final_ckpt}" "${attempt_log}"
      break
    else
      code=$?
      echo "[$(stamp)] ${variant}: attempt ${attempt} FAILED " \
        "(exit=${code}); ${attempt_log}" >&2
      record "${variant}" "${MODE}_attempt${attempt}" "failed" "${code}" \
        "${latest_ckpt}" "${attempt_log}"
    fi
  done

  if [[ "${success}" != "1" && "${CONTINUE_ON_FAILURE}" != "1" ]]; then
    echo "[$(stamp)] stopping after ${variant} failure" >&2
    exit 1
  fi
done

echo "[$(stamp)] sequence finished"
echo "[$(stamp)] summary=${SUMMARY}"
