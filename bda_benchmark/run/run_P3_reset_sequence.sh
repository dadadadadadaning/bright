#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-features}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash run/run_P3_reset_5090.sh preflight

if [[ "${MODE}" == "features" || "${MODE}" == "all" ]]; then
  for feature in a0_rgb a0_pseudo a0_rgb_pseudo b0 b1 b2; do
    echo "[$(date '+%F %T')] ${feature}: smoke"
    FEATURE_VARIANT="${feature}" bash run/run_P3_reset_5090.sh feature_smoke
    echo "[$(date '+%F %T')] ${feature}: controlled 1500-step audit"
    FEATURE_VARIANT="${feature}" bash run/run_P3_reset_5090.sh feature_train
  done
  bash run/run_P3_reset_5090.sh summarize_features
fi

if [[ "${MODE}" == "fusion" || "${MODE}" == "all" ]]; then
  DECISION="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_RESET}/FEATURE_AUDIT/feature_source_decision.json"
  test -s "${DECISION}" || { echo "Missing feature decision: ${DECISION}" >&2; exit 1; }
  SAR_SOURCE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "${DECISION}")"
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"], r["decision"]' "${DECISION}"
  VARIANT=c0 SAR_SOURCE=b0 OPTIMIZER_STEPS=0 bash run/run_P3_reset_5090.sh model_train
  VARIANT=c1 SAR_SOURCE="${SAR_SOURCE}" bash run/run_P3_reset_5090.sh model_smoke
  VARIANT=c1 SAR_SOURCE="${SAR_SOURCE}" bash run/run_P3_reset_5090.sh model_train
  C1_SUMMARY="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_RESET}/MODEL/C1_${SAR_SOURCE^^}_3000_seed${SEED:-3407}/training_summary.json"
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["advancement_gate"]["passed"], "C1 failed: do not run C2"' "${C1_SUMMARY}"
  VARIANT=c2 SAR_SOURCE="${SAR_SOURCE}" bash run/run_P3_reset_5090.sh model_smoke
  VARIANT=c2 SAR_SOURCE="${SAR_SOURCE}" bash run/run_P3_reset_5090.sh model_train
fi

if [[ "${MODE}" == "severity" ]]; then
  DECISION="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_RESET}/FEATURE_AUDIT/feature_source_decision.json"
  test -s "${DECISION}" || { echo "Missing feature decision: ${DECISION}" >&2; exit 1; }
  SAR_SOURCE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "${DECISION}")"
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"], r["decision"]' "${DECISION}"
  echo "[$(date '+%F %T')] c1s/${SAR_SOURCE}: severity-only smoke"
  # Do not inherit the formal-screen step/evaluation settings.  Smoke always
  # performs exactly 20 optimiser steps and evaluates its two-batch subset at
  # step 20; the following model_train command owns the requested screen size.
  VARIANT=c1s SAR_SOURCE="${SAR_SOURCE}" OPTIMIZER_STEPS=20 EVAL_EVERY=20 \
    bash run/run_P3_reset_5090.sh model_smoke
  echo "[$(date '+%F %T')] c1s/${SAR_SOURCE}: controlled severity-only screen"
  VARIANT=c1s SAR_SOURCE="${SAR_SOURCE}" OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}" \
    bash run/run_P3_reset_5090.sh model_train
fi

if [[ "${MODE}" == "heads" ]]; then
  : "${SAR_SOURCE:?Set SAR_SOURCE to the admitted feature source}"
  : "${INTERACTION:?Set INTERACTION to the admitted c1/c2 fusion}"
  : "${FUSION_CKPT:?Set FUSION_CKPT to its best_event_worst.pth}"
  VARIANT=d1 bash run/run_P3_reset_5090.sh model_smoke
  VARIANT=d1 bash run/run_P3_reset_5090.sh model_train
  D1_SUMMARY="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_RESET}/MODEL/D1_${SAR_SOURCE^^}_3000_seed${SEED:-3407}/training_summary.json"
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["advancement_gate"]["passed"], "D1 failed: do not run D2"' "${D1_SUMMARY}"
  VARIANT=d2 bash run/run_P3_reset_5090.sh model_smoke
  VARIANT=d2 bash run/run_P3_reset_5090.sh model_train
fi
