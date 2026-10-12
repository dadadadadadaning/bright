#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_q_factorial_v1
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest \
      "$MODULE.test_diagnostics" \
      "$MODULE.test_compare" \
      "$MODULE.test_evaluate_q" \
      "$MODULE.test_q_losses" \
      "$MODULE.test_q_variants" \
      "$MODULE.test_summarize" \
      "$MODULE.test_paired_spatial" \
      "$MODULE.test_paired_prediction"
    "$PYTHON" -m "$MODULE.train_q" --help
    "$PYTHON" -m "$MODULE.evaluate_q" --help
    "$PYTHON" -m "$MODULE.audit_saved_q" --help
    "$PYTHON" -m "$MODULE.compare" --help
    "$PYTHON" -m "$MODULE.summarize" --help
    "$PYTHON" -m "$MODULE.paired_spatial" --help
    "$PYTHON" -m "$MODULE.paired_prediction" --help
    echo 'CPU/preflight only; no BRIGHT Train, Val or Test image was evaluated.'
    ;;
  source)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_SOURCE_OUTPUT:?Set a NEW output for original frozen Q/B Val450}"
    "$PYTHON" -u -m "$MODULE.evaluate_q" \
      --protocol "$E0_PROTOCOL" --source-front --output "$Q_SOURCE_OUTPUT"
    ;;
  smoke_source)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_SOURCE_OUTPUT:?Set a NEW smoke output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate_q" \
      --protocol "$E0_PROTOCOL" --source-front --output "$Q_SOURCE_OUTPUT" \
      --limit "${Q_EVAL_LIMIT:-2}"
    ;;
  train)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_TRAIN_ARM:?Set one Q arm, e.g. baseline or semantic_reconstruction}"
    : "${Q_TRAIN_OUTPUT:?Set a NEW Train output directory}"
    "$PYTHON" -u -m "$MODULE.train_q" \
      --protocol "$E0_PROTOCOL" --arm "$Q_TRAIN_ARM" --output "$Q_TRAIN_OUTPUT"
    ;;
  smoke_train)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_TRAIN_ARM:?Set one Q arm}"
    : "${Q_TRAIN_OUTPUT:?Set a NEW smoke Train output directory}"
    "$PYTHON" -u -m "$MODULE.train_q" \
      --protocol "$E0_PROTOCOL" --arm "$Q_TRAIN_ARM" \
      --output "$Q_TRAIN_OUTPUT" --steps 2 --eval_every 2 --smoke
    ;;
  evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_EVAL_CHECKPOINT:?Set a completed formal Train last.pth}"
    : "${Q_EVAL_OUTPUT:?Set a NEW Val450 output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate_q" \
      --protocol "$E0_PROTOCOL" --checkpoint "$Q_EVAL_CHECKPOINT" \
      --output "$Q_EVAL_OUTPUT"
    ;;
  smoke_evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native all-path E0 directory}"
    : "${Q_EVAL_CHECKPOINT:?Set a completed Q Train last.pth}"
    : "${Q_EVAL_OUTPUT:?Set a NEW Val smoke output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate_q" \
      --protocol "$E0_PROTOCOL" --checkpoint "$Q_EVAL_CHECKPOINT" \
      --output "$Q_EVAL_OUTPUT" --limit "${Q_EVAL_LIMIT:-2}"
    ;;
  audit)
    : "${Q_AUDIT_SAMEPASS:?Set a completed source or trained-Q evaluate_q output}"
    : "${Q_AUDIT_OUTPUT:?Set a NEW read-only Q/B diagnostic output}"
    "$PYTHON" -u -m "$MODULE.audit_saved_q" \
      --samepass "$Q_AUDIT_SAMEPASS" --output "$Q_AUDIT_OUTPUT"
    ;;
  smoke_audit)
    : "${Q_AUDIT_SAMEPASS:?Set a completed short same-pass output}"
    : "${Q_AUDIT_OUTPUT:?Set a NEW smoke diagnostic output}"
    "$PYTHON" -u -m "$MODULE.audit_saved_q" \
      --samepass "$Q_AUDIT_SAMEPASS" --output "$Q_AUDIT_OUTPUT" --smoke
    ;;
  summarize)
    : "${Q_AUDIT_OUTPUT:?Set a completed full-Val frozen-Q audit directory}"
    "$PYTHON" -m "$MODULE.summarize" --audit "$Q_AUDIT_OUTPUT"
    ;;
  spatial)
    : "${Q_SPATIAL_SAMEPASS:?Set a completed original Q source Val450 directory}"
    : "${Q_SPATIAL_OUTPUT:?Set a NEW paired spatial diagnostic output}"
    "$PYTHON" -u -m "$MODULE.paired_spatial" \
      --samepass "$Q_SPATIAL_SAMEPASS" --output "$Q_SPATIAL_OUTPUT"
    ;;
  smoke_spatial)
    : "${Q_SPATIAL_SAMEPASS:?Set a completed short same-pass output}"
    : "${Q_SPATIAL_OUTPUT:?Set a NEW smoke paired spatial output}"
    "$PYTHON" -u -m "$MODULE.paired_spatial" \
      --samepass "$Q_SPATIAL_SAMEPASS" --output "$Q_SPATIAL_OUTPUT" --smoke
    ;;
  paired_prediction)
    : "${Q_COMPARE_BASELINE:?Set a completed baseline Q Val450 evaluation}"
    : "${Q_COMPARE_CANDIDATE:?Set one completed candidate Q Val450 evaluation}"
    : "${Q_PAIRED_OUTPUT:?Set a NEW paired-prediction output}"
    "$PYTHON" -u -m "$MODULE.paired_prediction" \
      --baseline "$Q_COMPARE_BASELINE" \
      --candidate "$Q_COMPARE_CANDIDATE" \
      --output "$Q_PAIRED_OUTPUT"
    ;;
  compare)
    : "${Q_COMPARE_BASELINE:?Set the completed baseline Q Val450 evaluation}"
    : "${Q_COMPARE_CANDIDATE:?Set one completed candidate Q Val450 evaluation}"
    : "${Q_COMPARE_OUTPUT:?Set a NEW comparison output}"
    "$PYTHON" -m "$MODULE.compare" \
      --baseline "$Q_COMPARE_BASELINE" \
      --candidate "$Q_COMPARE_CANDIDATE" \
      --output "$Q_COMPARE_OUTPUT"
    ;;
  *)
    echo 'Use preflight|source|smoke_source|train|smoke_train|evaluate|smoke_evaluate|audit|smoke_audit|summarize|spatial|smoke_spatial|paired_prediction|compare' >&2
    exit 2
    ;;
esac
