#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
MODULE=bda_benchmark.experiments.damageformer_partition_learning_v1
PACKAGE_DIR="$ROOT/bda_benchmark/experiments/damageformer_partition_learning_v1"
MODE="${1:-preflight}"

if (( $# > 1 )); then
  echo 'Supply one mode only; configure paths through environment variables.' >&2
  exit 2
fi
case "$MODE" in
  version|preflight|report|smoke_structure|structure|smoke_multimodal|multimodal|smoke_support|support) ;;
  *) echo 'Use version|preflight|smoke_structure|structure|smoke_multimodal|multimodal|smoke_support|support|report' >&2; exit 2 ;;
esac
: "${PYTHON:?Set an absolute Python executable path}"
if [[ "$PYTHON" != /* || ! -x "$PYTHON" ]]; then
  echo "PYTHON must be an absolute executable path: $PYTHON" >&2
  exit 2
fi

# Check the actual imported source before any model/data loading. Keep report
# JSON-only, and allow version/preflight without checkpoint/output variables.
if [[ "$MODE" != report ]]; then
  "$PYTHON" - "$PACKAGE_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

expected_dir = Path(sys.argv[1]).resolve()
print('PYTHON:', sys.executable, flush=True)
try:
    import bda_benchmark.experiments.damageformer_partition_learning_v1.train as training
    import bda_benchmark.experiments.damageformer_partition_learning_v1.reproduction as reproduction
except Exception as exc:
    raise SystemExit(
        'INSTALL CHECK FAILED: cannot import the complete updated package: '
        + type(exc).__name__ + ': ' + str(exc)
    ) from exc

for module, filename in ((training, 'train.py'), (reproduction, 'reproduction.py')):
    actual_path = Path(module.__file__).resolve()
    print('IMPORTED SOURCE:', actual_path, flush=True)
    if actual_path != expected_dir / filename:
        raise SystemExit('INSTALL CHECK FAILED: imported source is outside this package directory')
    print('SOURCE SHA256:', hashlib.sha256(actual_path.read_bytes()).hexdigest(), flush=True)

if not callable(getattr(training, 'initial_gate', None)):
    raise SystemExit('INSTALL CHECK FAILED: old train.py; missing initial_gate')
if not callable(getattr(training, 'reproduction_policy', None)):
    raise SystemExit('INSTALL CHECK FAILED: old train.py; missing reproduction_policy')
main_code = getattr(training.main, '__code__', None)
if (main_code is None or 'history_bounded_v1' not in main_code.co_consts
        or 'initial_gate' not in main_code.co_names or 'reproduction_policy' not in main_code.co_names):
    raise SystemExit('INSTALL CHECK FAILED: main does not use history_bounded_v1')
for name in ('load_history', 'build_policy', 'compare_baselines', 'enforce', 'json_safe_evidence'):
    if not callable(getattr(reproduction, name, None)):
        raise SystemExit('INSTALL CHECK FAILED: incomplete reproduction.py; missing ' + name)
print('VERSION CHECK PASSED: history_bounded_v1', flush=True)
PY
fi
if [[ "$MODE" == version ]]; then
  exit 0
fi

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest discover -s "$PACKAGE_DIR" -t "$ROOT" -p 'test_*.py' -v
  "$PYTHON" -m "$MODULE.train" --help
  exit 0
fi
: "${OUTPUT_ROOT:?Set the output directory}"
if [[ "$MODE" == report ]]; then
  "$PYTHON" -m "$MODULE.report" --root "$OUTPUT_ROOT"
  exit 0
fi

: "${FRONT_CKPT:?Set the immutable front snapshot matching the source run}"
: "${SOURCE_RUN:?Set a completed formal ROI_EXCHANGE_V1 training directory}"
for INPUT in "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/manifest.json" "$SOURCE_RUN/summary.json"; do
  if [[ ! -s "$INPUT" ]]; then
    echo "Required input missing or empty: $INPUT" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]]; then
  echo "Output exists; choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

SUITE="${MODE#smoke_}"
COMMON=(--suite "$SUITE" --checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN"
  --source_head hard --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT" --batch "${BATCH:-2}" --accum "${ACCUM:-8}"
  --workers "${WORKERS:-4}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}"
  --device "${DEVICE:-cuda:0}" --structure_weight "${STRUCTURE_WEIGHT:-0.1}"
  --sar_weight "${SAR_WEIGHT:-0.2}" --support_temperature "${SUPPORT_TEMPERATURE:-0.2}")
# Stale VARIANTS, EXP, STEPS and EVAL_EVERY never select groups or alter this protocol.
if [[ "$MODE" == smoke_* ]]; then
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke
else
  "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" --steps 5000 --eval_every 500
fi
"$PYTHON" -m "$MODULE.report" --root "$OUTPUT_ROOT"
