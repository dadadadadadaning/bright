#!/usr/bin/env bash
# One-command H12-D1 training: preflight, D0 guard, D1 smoke, D1 1500-step screen.
# This does not access the test set.  Set SKIP_AUDIT=0 to run the val audit first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANT="${VARIANT:-d1}"
STEPS="${OPTIMIZER_STEPS:-1500}"
SKIP_AUDIT="${SKIP_AUDIT:-1}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H12_DGB}"
mkdir -p "${LOG_ROOT}"

export VARIANTS="${VARIANT}"
export OPTIMIZER_STEPS="${STEPS}"
export SKIP_AUDIT
export SEED
export CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

exec bash run/run_H12_DGB_sequence.sh screen
