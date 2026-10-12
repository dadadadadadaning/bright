#!/usr/bin/env bash
set -euo pipefail

# One entry point for the native-SAR diagnosis chain.
# It performs validation-only probes and controlled training.  Test is never
# evaluated or used for checkpoint/candidate selection.

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
STEPS="${OPTIMIZER_STEPS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
EVAL_ROOT="${EVAL_ROOT:-/root/shared-nvme/BRIGHT-eval-results/H6_SDCR}"
PROBE_JSON="${PROBE_JSON:-${EVAL_ROOT}/H6_A2_SAR_stage_probe_val.json}"
REPORT_JSON="${REPORT_JSON:-${OUTPUT_ROOT}/H6_A2_core_diagnosis${STEPS}_seed${SEED}.json}"
mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}" "${OUTPUT_ROOT}"

stamp() { date '+%F %T'; }
announce() { echo "[$(stamp)] $*"; }

case "${MODE}" in
  preflight)
    bash run/run_H6_SDCR_5090.sh preflight
    bash run/run_H6_SAR_stage_probe.sh preflight
    python -m py_compile script/standard_ML/summarize_H6_A2_core_diagnosis.py
    bash -n run/run_H6_A2_freeze_ablation.sh
    bash -n run/run_H6_A2_core_diagnosis.sh
    announce "H6-A2 core diagnosis preflight passed"
    exit 0
    ;;
  smoke)
    announce "SAR stage probe smoke starts"
    OUTPUT_JSON="${EVAL_ROOT}/H6_A2_SAR_stage_probe_smoke.json" \
      bash run/run_H6_SAR_stage_probe.sh smoke
    announce "F0/F1/F2 smoke starts"
    OPTIMIZER_STEPS=20 EVAL_EVERY=20 \
      bash run/run_H6_A2_freeze_ablation.sh smoke
    announce "H6-A2 core diagnosis smoke passed"
    exit 0
    ;;
  report)
    announce "strict independent validation of existing F0/F1/F2 starts"
    OPTIMIZER_STEPS="${STEPS}" SEED="${SEED}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
      bash run/run_H6_A2_freeze_ablation.sh evaluate
    python script/standard_ML/summarize_H6_A2_core_diagnosis.py \
      --output_root "${OUTPUT_ROOT}" --steps "${STEPS}" --seed "${SEED}" \
      --probe_json "${PROBE_JSON}" --output_json "${REPORT_JSON}"
    announce "report repaired; report=${REPORT_JSON}; test accessed=NEVER"
    exit 0
    ;;
  screen) ;;
  *) echo "Expected preflight|smoke|screen|report" >&2; exit 2 ;;
esac

announce "shared preflight starts"
bash run/run_H6_A2_core_diagnosis.sh preflight

announce "full validation-only SAR stage probe starts"
OUTPUT_JSON="${PROBE_JSON}" \
  bash run/run_H6_SAR_stage_probe.sh val \
  2>&1 | tee "${LOG_ROOT}/H6_A2_SAR_stage_probe_val.log"

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  announce "controlled F0/F1/F2 smoke starts"
  OPTIMIZER_STEPS=20 EVAL_EVERY=20 \
    bash run/run_H6_A2_freeze_ablation.sh smoke \
    2>&1 | tee "${LOG_ROOT}/H6_A2_F0_F1_F2_smoke.log"
fi

announce "controlled F0/F1/F2 ${STEPS}-step screen starts"
OPTIMIZER_STEPS="${STEPS}" EVAL_EVERY="${EVAL_EVERY}" \
  SEED="${SEED}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  bash run/run_H6_A2_freeze_ablation.sh screen \
  2>&1 | tee "${LOG_ROOT}/H6_A2_F0_F1_F2_screen${STEPS}_seed${SEED}.log"

announce "strict independent validation of selected F0/F1/F2 weights starts"
OPTIMIZER_STEPS="${STEPS}" SEED="${SEED}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  bash run/run_H6_A2_freeze_ablation.sh evaluate

announce "validation-only decision report starts"
python script/standard_ML/summarize_H6_A2_core_diagnosis.py \
  --output_root "${OUTPUT_ROOT}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --probe_json "${PROBE_JSON}" \
  --output_json "${REPORT_JSON}"

announce "H6-A2 core diagnosis finished; report=${REPORT_JSON}; test accessed=NEVER"
