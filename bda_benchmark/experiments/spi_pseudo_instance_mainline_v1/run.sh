#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|selective-smoke|selective-audit|geometry-controls|rgid-train|rgid-audit|summarize) ;;
  *)
    echo "Choose preflight|selective-smoke|selective-audit|geometry-controls|rgid-train|rgid-audit|summarize" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n "${SCRIPT_DIR}/run.sh"
  python -m py_compile "${SCRIPT_DIR}/verify_stack.py" "${SCRIPT_DIR}/summarize_results.py"
  python -m bda_benchmark.experiments.spi_pseudo_instance_mainline_v1.verify_stack \
    --project-root "${PROJECT_ROOT}"
  bash experiments/spi_selective_context_v1/run.sh preflight
  bash experiments/spi_rgid_semantic_v1/run.sh preflight
  echo "SPI pseudo-instance main-line V1 preflight passed"
  exit 0
fi

case "${MODE}" in
  selective-smoke)
    exec bash experiments/spi_selective_context_v1/run.sh smoke
    ;;
  selective-audit)
    exec bash experiments/spi_selective_context_v1/run.sh audit
    ;;
  geometry-controls)
    exec bash experiments/spi_rgid_semantic_v1/run.sh controls
    ;;
  rgid-train)
    exec bash experiments/spi_rgid_semantic_v1/run.sh train-online
    ;;
  rgid-audit)
    : "${RGID_CKPT:?RGID_CKPT is required}"
    : "${RGID_PROVENANCE_JSON:?RGID_PROVENANCE_JSON is required}"
    exec bash experiments/spi_rgid_semantic_v1/run.sh rgid
    ;;
  summarize)
    : "${SELECTIVE_RESULT:?SELECTIVE_RESULT is required}"
    args=(--selective-json "${SELECTIVE_RESULT}" --iterations "${BOOTSTRAP_ITERATIONS:-10000}")
    if [[ -n "${GEOMETRY_RESULT:-}" ]]; then
      args+=(--geometry-json "${GEOMETRY_RESULT}")
    fi
    if [[ -n "${SUMMARY_OUTPUT:-}" ]]; then
      args+=(--output-json "${SUMMARY_OUTPUT}")
    fi
    exec python -m bda_benchmark.experiments.spi_pseudo_instance_mainline_v1.summarize_results "${args[@]}"
    ;;
esac

