#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)}"
SELF="$(cd "$(dirname "$0")" && pwd)"
for rel in \
  model/cmr_v1/replay.py \
  script/standard_ML/cmr_v1/build_token_cache.py \
  tests/cmr_v1/test_replay.py \
  run/run_CMR_V1_isolated.sh; do
  mkdir -p "${ROOT}/$(dirname "$rel")"
  cp -f "${SELF}/${rel}" "${ROOT}/${rel}"
done
echo "CMR-V1 replay hotfix applied to: ${ROOT}"
