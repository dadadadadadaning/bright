#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/6] Environment versions"
"$PYTHON_BIN" - <<'PY'
import platform
import numpy
import scipy
import skimage
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("scikit-image:", skimage.__version__)
PY

echo "[2/6] Install the isolated editable package"
"$PYTHON_BIN" -m pip install -e . --no-deps --no-build-isolation

echo "[3/6] Verify import path and model parameter count"
"$PYTHON_BIN" - <<'PY'
import rgid
from rgid import RGIDHead

print("RGID path:", rgid.__file__)
normalized = rgid.__file__.replace("\\", "/")
if "experiments/rgid_s9ic_clean/rgid" not in normalized:
    raise RuntimeError("RGID imported from wrong path: " + rgid.__file__)
model = RGIDHead()
print("RGID parameters:", sum(p.numel() for p in model.parameters()))
PY

echo "[4/6] Run only the isolated RGID unit tests"
"$PYTHON_BIN" -m pytest -q \
  tests/test_targets_losses.py \
  tests/test_geometry.py

echo "[5/6] Run the synthetic integration smoke test"
"$PYTHON_BIN" scripts/smoke_test_synthetic.py

echo "[6/6] Run the two-sample cached pipeline integration test"
"$PYTHON_BIN" -m pytest -q tests/test_cached_pipeline_integration.py

echo "All RGID basic tests passed."
