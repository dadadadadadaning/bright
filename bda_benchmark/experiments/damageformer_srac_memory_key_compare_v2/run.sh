#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_memory_key_compare_v2
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -m unittest discover -s "${MODULE//.//}" -t "$ROOT" -p 'test_*.py' -v
    for entry in build_keys diagnose train_correction evaluate_correction recheck_replay; do
      "$PYTHON" -m "$MODULE.$entry" --help
    done
    echo 'CPU synthetic contracts completed; no real BRIGHT Train/Val/Test or CUDA validation.'
    exit 0 ;;
  report|report_smoke)
    : "${KEY_OUTPUT_ROOT:?Set the same KEY_OUTPUT_ROOT used for this experiment}"
    target="$KEY_OUTPUT_ROOT/diagnose"
    if [[ "$MODE" == report_smoke ]]; then target="$KEY_OUTPUT_ROOT/diagnose_smoke"; fi
    "$PYTHON" -m "$MODULE.report" --input "$target"
    exit 0 ;;
  build_smoke|smoke|build|diagnose|train_smoke|train|evaluate) ;;
  *) echo 'Use preflight|build_smoke|smoke|build|diagnose|report_smoke|report|train_smoke|train|evaluate' >&2; exit 2 ;;
esac
: "${KEY_OUTPUT_ROOT:?Set a NEW experiment root outside all original runs}"
# Use the recorded actual night-run root; never guess the latest directory.
if [[ -z "${E0_PROTOCOL:-}" ]]; then
  detected_protocol="$("$PYTHON" - <<'PY'
from pathlib import Path
log = Path('/root/shared-nvme/bright_region_night_20261013_001703_3671794.log')
if not log.is_file():
    raise SystemExit('Missing recorded night log; export E0_PROTOCOL to the actual passed fresh E0 path.')
roots = list(dict.fromkeys(s.split('=', 1)[1].strip() for s in log.read_text(errors='replace').splitlines()
                          if s.startswith('OUTPUT_ROOT=')))
if len(roots) != 1:
    raise SystemExit('Expected one actual OUTPUT_ROOT= in the log; export E0_PROTOCOL explicitly.')
root = Path(roots[0])
if not root.is_absolute():
    raise SystemExit('Recorded OUTPUT_ROOT is not absolute.')
protocol = root / 'fresh_e0/protocol.json'
if not protocol.is_file():
    raise SystemExit('Fresh E0 protocol is not available; do not start another queue.')
print(protocol)
PY
  )"
  export E0_PROTOCOL="$detected_protocol"
fi
# Reject missing/failed/smoke protocols before creating any experiment directory.
"$PYTHON" -c 'import sys; from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import protocol_record; protocol_record(sys.argv[1]); print("Using passed E0:", sys.argv[1])' "$E0_PROTOCOL"
COMMON=(--protocol "$E0_PROTOCOL")
case "$MODE" in
  build_smoke)
    "$PYTHON" -u -m "$MODULE.build_keys" "${COMMON[@]}" --output "$KEY_OUTPUT_ROOT/keys_smoke" --limit-images 4 ;;
  smoke)
    "$PYTHON" -u -m "$MODULE.diagnose" "${COMMON[@]}" --keys "$KEY_OUTPUT_ROOT/keys_smoke" --output "$KEY_OUTPUT_ROOT/diagnose_smoke" --limit 2 ;;
  build)
    "$PYTHON" -u -m "$MODULE.build_keys" "${COMMON[@]}" --output "$KEY_OUTPUT_ROOT/keys" ;;
  diagnose)
    "$PYTHON" -u -m "$MODULE.diagnose" "${COMMON[@]}" --keys "$KEY_OUTPUT_ROOT/keys" --output "$KEY_OUTPUT_ROOT/diagnose" --policy both ;;
  train_smoke)
    "$PYTHON" -u -m "$MODULE.train_correction" "${COMMON[@]}" --keys "$KEY_OUTPUT_ROOT/keys" --output "$KEY_OUTPUT_ROOT/correction_smoke" --smoke --steps 20 ;;
  train)
    "$PYTHON" -u -m "$MODULE.train_correction" "${COMMON[@]}" --keys "$KEY_OUTPUT_ROOT/keys" --diagnostics "$KEY_OUTPUT_ROOT/diagnose" --output "$KEY_OUTPUT_ROOT/correction_fit" --steps 5000 ;;
  evaluate)
    "$PYTHON" -u -m "$MODULE.evaluate_correction" "${COMMON[@]}" --keys "$KEY_OUTPUT_ROOT/keys" --fit "$KEY_OUTPUT_ROOT/correction_fit" --output "$KEY_OUTPUT_ROOT/correction_independent_val" --interventions ;;
esac
