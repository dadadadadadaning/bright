#!/usr/bin/env bash
set -euo pipefail

# Controlled E1-E3 chain.  Every candidate starts from the exact same
# N1-shared Heads500 checkpoint.  Joint Stage2 adaptation and test are locked.

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HSDNET}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/OA_HSDNET}"
BASE_HEADS="${BASE_HEADS:-${OUTPUT_ROOT}/OA_HSDNET_N1_SHARED_heads500_seed${SEED}/best_event_worst.pth}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${OUTPUT_ROOT}/OA_HSDNET_N1_SHARED_deep1500_seed${SEED}/training_summary.json}"
STEPS="${OPTIMIZER_STEPS:-1500}"
mkdir -p "${LOG_ROOT}"

case "${MODE}" in
  preflight)
    bash run/run_OABH_5090.sh preflight
    exit 0
    ;;
  e1|e2|e3|all) ;;
  *) echo "Usage: $0 preflight|e1|e2|e3|all" >&2; exit 2 ;;
esac

test -s "${BASE_HEADS}" || {
  echo "Missing shared Heads500 checkpoint: ${BASE_HEADS}" >&2
  exit 1
}

bash run/run_OABH_5090.sh preflight

run_candidate() {
  local code="$1"
  local style_mode="$2"
  local upper="${code^^}"
  local smoke_name="OA_HSDNET_N1_SHARED_${upper}_smoke20_seed${SEED}"
  local exp_name="OA_HSDNET_N1_SHARED_${upper}_deep${STEPS}_seed${SEED}"
  local exp_dir="${OUTPUT_ROOT}/${exp_name}"

  echo "[$(date '+%F %T')] ${code}: smoke | style=${style_mode}"
  EVENT_BALANCED_SAMPLING=1 \
  OABH_SAR_STYLE_MODE="${style_mode}" \
  OABH_SAR_MIX_PROBABILITY="${SAR_MIX_PROBABILITY:-0.5}" \
  OABH_SAR_MIX_ALPHA="${SAR_MIX_ALPHA:-0.1}" \
  OABH_FULL_INIT_CKPT="${BASE_HEADS}" \
  VARIANT=n1_shared SEED="${SEED}" EXP_NAME="${smoke_name}" \
    bash run/run_OABH_5090.sh smoke \
    > "${LOG_ROOT}/${smoke_name}.log" 2>&1

  echo "[$(date '+%F %T')] ${code}: controlled event-balanced Deep-only"
  EVENT_BALANCED_SAMPLING=1 \
  OABH_SAR_STYLE_MODE="${style_mode}" \
  OABH_SAR_MIX_PROBABILITY="${SAR_MIX_PROBABILITY:-0.5}" \
  OABH_SAR_MIX_ALPHA="${SAR_MIX_ALPHA:-0.1}" \
  OABH_FULL_INIT_CKPT="${BASE_HEADS}" \
  VARIANT=n1_shared SEED="${SEED}" EXP_NAME="${exp_name}" \
  OPTIMIZER_STEPS="${STEPS}" EVAL_EVERY="${EVAL_EVERY:-500}" \
    bash run/run_OABH_5090.sh deep \
    > "${LOG_ROOT}/${exp_name}.log" 2>&1

  test -s "${exp_dir}/training_summary.json" || {
    echo "Missing training summary: ${exp_dir}/training_summary.json" >&2
    return 1
  }
  if [[ -s "${BASELINE_SUMMARY}" ]]; then
    python script/standard_ML/summarize_OABH_event_style.py \
      --baseline "${BASELINE_SUMMARY}" \
      --candidate "${exp_dir}/training_summary.json" \
      --output "${exp_dir}/${code}_advancement_gate.json"
  fi
}

if [[ "${MODE}" == "e1" || "${MODE}" == "all" ]]; then
  run_candidate e1 none
fi
if [[ "${MODE}" == "e2" || "${MODE}" == "all" ]]; then
  run_candidate e2 mixstyle
fi
if [[ "${MODE}" == "e3" || "${MODE}" == "all" ]]; then
  run_candidate e3 adapter
fi

echo "OA-HSDNet E1-E3 controlled sequence completed."
echo "All candidates used one Heads500 checkpoint; Joint and Test were not run."
