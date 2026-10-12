#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BDA_DIR="$ROOT_DIR/bda_benchmark"
MODE="${1:-preflight}"

cd "$BDA_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

preflight() {
  python -m unittest -v tests.test_multimodal_damage_stages
  python script/standard_ML/audit_multimodal_damage_dependencies.py
}

require_source() {
  local path="$1"
  local name="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing official ${name} source: ${path}" >&2
    echo "third_party code is never replaced by a local approximation." >&2
    exit 2
  fi
}

case "$MODE" in
  preflight|audit)
    preflight
    ;;
  r1)
    preflight
    require_source "$ROOT_DIR/third_party/FlowMamba/changedetection" "FlowMamba"
    echo "FlowMamba source is ready; run entry remains locked until its exact adapter preflight passes."
    ;;
  r2|r3)
    preflight
    require_source "$ROOT_DIR/third_party/M2CD/opencd" "M2CD"
    echo "M2CD source is ready; run entry remains locked until its exact adapter preflight passes."
    ;;
  r4|r5|r6)
    preflight
    require_source "$ROOT_DIR/third_party/M2CD/opencd" "M2CD"
    require_source "$ROOT_DIR/third_party/MM-OVSeg" "MM-OVSeg/SAR-DINO"
    if [[ ! -s "${SAR_DINO_CKPT:-}" ]]; then
      echo "Missing SAR_DINO_CKPT; R4 probe must pass before R4-R6 are admitted." >&2
      exit 2
    fi
    echo "R4-R6 dependencies found; the SAR-DINO probe remains the mandatory gate."
    ;;
  *)
    echo "Usage: $0 {preflight|audit|r1|r2|r3|r4|r5|r6}" >&2
    exit 2
    ;;
esac
