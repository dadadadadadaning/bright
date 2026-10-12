#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-baselines}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/ExternalCD_BRIGHT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/ExternalCD_BRIGHT}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONTROLLER_LOG="${LOG_ROOT}/controller_${MODE}_${TIMESTAMP}.log"
REPORT_TSV="${LOG_ROOT}/controller_${MODE}_${TIMESTAMP}.tsv"
FAILURES=0

exec > >(tee -a "${CONTROLLER_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

announce() {
  echo "[$(timestamp)] $*"
}

record_status() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "$5" \
    >> "${REPORT_TSV}"
}

run_checked() {
  local label="$1"
  shift
  local attempt=0
  local status=1
  while (( attempt <= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    announce "${label}: attempt ${attempt} starts"
    if "$@"; then
      announce "${label}: passed"
      status=0
      break
    fi
    announce "${label}: attempt ${attempt} failed"
  done
  if (( status != 0 )); then
    FAILURES=$((FAILURES + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      announce "sequence stops after ${label} failure"
      exit 1
    fi
  fi
  return "${status}"
}

run_experiment() {
  local variant="$1"
  local suffix="$2"
  local fca_weight="$3"
  local smoke_name="ExternalCD_${variant^^}_${suffix}_smoke20_seed${SEED}"
  local pilot_name="ExternalCD_${variant^^}_${suffix}_pilot${OPTIMIZER_STEPS}_seed${SEED}"
  local smoke_log="${LOG_ROOT}/${smoke_name}.log"
  local pilot_log="${LOG_ROOT}/${pilot_name}.log"

  announce "${variant}/${suffix}: smoke starts"
  if env \
      VARIANT="${variant}" \
      FCA_WEIGHT="${fca_weight}" \
      EXP_NAME="${smoke_name}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SAR_STATS_JSON="${SAR_STATS_JSON}" \
      bash run/run_ExternalCD_BRIGHT_5090.sh smoke \
      > "${smoke_log}" 2>&1; then
    announce "${variant}/${suffix}: smoke passed"
    record_status "${variant}" "${suffix}" "smoke" "PASS" "${smoke_log}"
  else
    announce "${variant}/${suffix}: smoke failed; see ${smoke_log}"
    record_status "${variant}" "${suffix}" "smoke" "FAIL" "${smoke_log}"
    FAILURES=$((FAILURES + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    return 1
  fi

  local attempt=0
  local trained=0
  while (( attempt <= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    announce "${variant}/${suffix}: ${OPTIMIZER_STEPS}-step pilot attempt ${attempt} starts"
    if env \
        VARIANT="${variant}" \
        FCA_WEIGHT="${fca_weight}" \
        EXP_NAME="${pilot_name}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        SAR_STATS_JSON="${SAR_STATS_JSON}" \
        OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" \
        EVAL_EVERY="${EVAL_EVERY}" \
        SAVE_EVERY="${SAVE_EVERY}" \
        bash run/run_ExternalCD_BRIGHT_5090.sh pilot \
        > "${pilot_log}" 2>&1; then
      trained=1
      break
    fi
    announce "${variant}/${suffix}: pilot attempt ${attempt} failed; retry uses latest_resume if present"
  done

  if (( trained == 0 )); then
    FAILURES=$((FAILURES + 1))
    record_status "${variant}" "${suffix}" "pilot" "FAIL" "${pilot_log}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    return 1
  fi

  local summary="${OUTPUT_ROOT}/${pilot_name}/training_summary.json"
  if [[ ! -s "${summary}" ]]; then
    announce "${variant}/${suffix}: training command ended but summary is missing: ${summary}"
    FAILURES=$((FAILURES + 1))
    record_status "${variant}" "${suffix}" "pilot" "NO_SUMMARY" "${pilot_log}"
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    return 1
  fi

  python - "${summary}" "${REPORT_TSV}" "${variant}" "${suffix}" \
    "${OUTPUT_ROOT}/${pilot_name}" "${pilot_log}" <<'PY'
import json
import sys

s = json.load(open(sys.argv[1], encoding="utf-8"))
m = s["best_metrics"]
def percent(key):
    value = m.get(key)
    return "NA" if value is None else f"{100 * float(value):.6f}"

selected = s.get("selected_step")
row = [
    sys.argv[3],
    sys.argv[4],
    "pilot",
    "PASS",
    str(s.get("completed_step", "NA")),
    str(selected if selected is not None else "NA"),
    percent("valid_miou"),
    percent("miou4"),
    percent("damaged_iou"),
    percent("destroyed_iou"),
    percent("loc_f1"),
    percent("event_mean_official_style_4class_miou"),
    percent("event_worst_official_style_4class_miou"),
    sys.argv[5],
    sys.argv[6],
]
with open(sys.argv[2], "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")

worst = m.get("event_worst_official_style_4class_miou")
worst_text = "NA" if worst is None else f"{100 * worst:.4f}"
print(
    "completed={completed} | valid={valid:.4f} | mIoU4={miou4:.4f} | "
    "Dmg={dmg:.4f} | Des={des:.4f} | Event={event:.4f} | Worst={worst}".format(
        completed=s.get("completed_step"),
        valid=100 * m["valid_miou"],
        miou4=100 * m["miou4"],
        dmg=100 * m["damaged_iou"],
        des=100 * m["destroyed_iou"],
        event=100 * m["event_mean_official_style_4class_miou"],
        worst=worst_text,
    )
)
PY
  announce "${variant}/${suffix}: pilot completed"
}

case "${MODE}" in
  x1)
    RUN_X1=1; RUN_X2=0; RUN_FCA=0 ;;
  baselines)
    RUN_X1=1; RUN_X2=1; RUN_FCA=0 ;;
  fca)
    RUN_X1=0; RUN_X2=0; RUN_FCA=1 ;;
  all)
    RUN_X1=1; RUN_X2=1; RUN_FCA=1 ;;
  *)
    echo "Usage: $0 x1|baselines|fca|all" >&2
    exit 2
    ;;
esac

printf 'variant\tconfiguration\tstage\tstatus\tcompleted_step\tselected_step\tvalid_miou_pp\tmiou4_pp\tdamaged_iou_pp\tdestroyed_iou_pp\tloc_f1_pp\tevent_mean_pp\tevent_worst_pp\toutput_dir\tlog\n' > "${REPORT_TSV}"

announce "shared preflight starts"
run_checked "shared preflight" bash run/run_ExternalCD_BRIGHT_5090.sh preflight

if [[ ! -s "${SAR_STATS_JSON}" ]]; then
  announce "train-only SAR statistics are missing; computing once"
  run_checked "SAR statistics" env \
    SAR_STATS_JSON="${SAR_STATS_JSON}" \
    bash run/run_HeterogeneousDamageFormer_5090.sh stats
else
  announce "frozen train-only SAR statistics found: ${SAR_STATS_JSON}"
fi

if (( RUN_X1 == 1 )); then
  run_experiment x1 plain 0 || true
fi

if (( RUN_X2 == 1 || RUN_FCA == 1 )); then
  HETECD_FILE="${REPOSITORY_ROOT}/third_party/HeteCD/models/hetecd.py"
  if [[ ! -s "${HETECD_FILE}" ]]; then
    announce "official HeteCD source missing: ${HETECD_FILE}"
    announce "clone/upload the official repository before running x2/fca"
    FAILURES=$((FAILURES + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
  else
    run_checked "official HeteCD adapter preflight" env \
      CHECK_HETECD=1 bash run/run_ExternalCD_BRIGHT_5090.sh preflight
    if (( RUN_X2 == 1 )); then
      run_experiment x2 noFCA 0 || true
    fi
    if (( RUN_FCA == 1 )); then
      run_experiment x2 FCA001 "${FCA_WEIGHT:-0.01}" || true
    fi
  fi
fi

announce "ExternalCD sequence finished; failures=${FAILURES}"
announce "controller log=${CONTROLLER_LOG}"
announce "report=${REPORT_TSV}"
exit "${FAILURES}"
