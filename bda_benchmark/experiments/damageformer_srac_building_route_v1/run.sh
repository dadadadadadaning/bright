#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_building_route_v1
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_train" "$MODULE.test_routing" "$MODULE.test_model"
    "$PYTHON" -m "$MODULE.train" --help
    "$PYTHON" -m "$MODULE.evaluate" --help
    "$PYTHON" -m "$MODULE.report" --help
    "$PYTHON" -m "$MODULE.cross_q" --help
    echo 'Preflight only; no Train, Val or Test images were read.'
    ;;
  smoke)
    : "${E0_PROTOCOL:?Set passed full legacy/native E0 protocol directory}"
    : "${ROUTE_FIT:?Set NEW smoke Train output directory}"
    extra=()
    if [[ -n "${ROUTE_Q_CHECKPOINT:-}" ]]; then extra+=(--q-checkpoint "$ROUTE_Q_CHECKPOINT"); fi
    "$PYTHON" -u -m "$MODULE.train" --protocol "$E0_PROTOCOL" \
      --output "$ROUTE_FIT" --steps 2 --eval-every 2 --smoke "${extra[@]}"
    ;;
  train)
    : "${E0_PROTOCOL:?Set passed full legacy/native E0 protocol directory}"
    : "${ROUTE_FIT:?Set NEW Train output directory}"
    extra=()
    if [[ -n "${ROUTE_Q_CHECKPOINT:-}" ]]; then extra+=(--q-checkpoint "$ROUTE_Q_CHECKPOINT"); fi
    "$PYTHON" -u -m "$MODULE.train" --protocol "$E0_PROTOCOL" \
      --output "$ROUTE_FIT" --steps "${ROUTE_STEPS:-2000}" \
      --eval-every "${ROUTE_EVAL_EVERY:-500}" \
      --threshold "${ROUTE_THRESHOLD:-0.5}" --soft-floor "${ROUTE_SOFT_FLOOR:-0.0}" \
      "${extra[@]}"
    ;;
  evaluate)
    : "${E0_PROTOCOL:?Set passed full legacy/native E0 protocol directory}"
    : "${ROUTE_FIT:?Set a completed Train route directory}"
    : "${ROUTE_EVAL:?Set a NEW Val450 output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate" --protocol "$E0_PROTOCOL" \
      --fit "$ROUTE_FIT" --output "$ROUTE_EVAL"
    ;;
  report)
    : "${ROUTE_EVAL:?Set a completed independent Val450 evaluation directory}"
    "$PYTHON" -m "$MODULE.report" --evaluation "$ROUTE_EVAL"
    ;;
  cross_q)
    : "${ROUTE_ORIGINAL_FIT:?Set completed original-Q route Train directory}"
    : "${ROUTE_ORIGINAL_EVAL:?Set completed original-Q route Val directory}"
    : "${ROUTE_BUILDING_FIT:?Set completed building-Q route Train directory}"
    : "${ROUTE_BUILDING_EVAL:?Set completed building-Q route Val directory}"
    "$PYTHON" -m "$MODULE.cross_q" \
      --original-fit "$ROUTE_ORIGINAL_FIT" --original-evaluation "$ROUTE_ORIGINAL_EVAL" \
      --building-fit "$ROUTE_BUILDING_FIT" --building-evaluation "$ROUTE_BUILDING_EVAL"
    ;;
  *)
    echo 'Use preflight|smoke|train|evaluate|report|cross_q' >&2
    exit 2
    ;;
esac
