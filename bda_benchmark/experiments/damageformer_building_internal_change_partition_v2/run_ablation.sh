#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-stage0}"
for V in p_s_struct_only p_c_change_only p_sc_residual; do
  echo "===== ${MODE} ${V} ====="
  bash "${HERE}/run.sh" "${MODE}" "${V}"
done
