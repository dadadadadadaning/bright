#!/usr/bin/env bash
set -euo pipefail

# Distribution-robust retraining protocol for the native-SAR A2 expert.
# This deliberately changes the training protocol, not the network topology:
#   1) square-root event balancing instead of full inverse-frequency sampling;
#   2) moderate damage-focused crops, biased toward the weak Damaged class;
#   3) three validation-only checkpoint selectors (pixel/event/damage balance).
# The test split is disabled until one candidate has been locked on validation.

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VARIANT=a2
export EVENT_BALANCE_POWER="${EVENT_BALANCE_POWER:-0.50}"
export FOCUS_PROBABILITY="${FOCUS_PROBABILITY:-0.35}"
export FOCUS_BACKGROUND_WEIGHT="${FOCUS_BACKGROUND_WEIGHT:-0}"
export FOCUS_INTACT_WEIGHT="${FOCUS_INTACT_WEIGHT:-1}"
export FOCUS_DAMAGED_WEIGHT="${FOCUS_DAMAGED_WEIGHT:-3}"
export FOCUS_DESTROYED_WEIGHT="${FOCUS_DESTROYED_WEIGHT:-2}"
export SAVE_SECONDARY_BEST=1
export ENABLE_TEST_MONITOR=0

case "${MODE}" in
  preflight|guard|smoke|pilot|full|eval_val|eval_test|summary) ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|full|eval_val|eval_test|summary" >&2
    exit 2
    ;;
esac

if [[ -z "${EXP_NAME:-}" && "${MODE}" != "preflight" ]]; then
  case "${MODE}" in
    guard) STEPS=0 ;;
    smoke) STEPS=20 ;;
    pilot) STEPS="${OPTIMIZER_STEPS:-3000}" ;;
    full) STEPS="${OPTIMIZER_STEPS:-8000}" ;;
    *) STEPS="selected" ;;
  esac
  export EXP_NAME="H6_A2_DG_${MODE}${STEPS}_seed${SEED:-3407}"
fi

exec bash "${SCRIPT_DIR}/run_H6_SDCR_5090.sh" "${MODE}"
