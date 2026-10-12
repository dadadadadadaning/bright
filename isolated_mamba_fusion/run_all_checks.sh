#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "isolated_mamba_fusion: run all checks"
echo "============================================================"

if [[ ! -d "isolated_mamba_fusion" ]]; then
  echo "[ERROR] Please run this script from the project root directory."
  echo "        Expected directory: isolated_mamba_fusion/"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[INFO] Current path:"
pwd
echo "[OK] Current path printed"

echo "------------------------------------------------------------"
echo "[INFO] Python version:"
"$PYTHON_BIN" --version
echo "[OK] Python is available"

echo "------------------------------------------------------------"
echo "[INFO] PyTorch / CUDA:"
"$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"[ERROR] Failed to import torch: {type(exc).__name__}: {exc}")
    raise

print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
PY
echo "[OK] PyTorch environment check passed"

echo "------------------------------------------------------------"
echo "[INFO] Syntax check: python -m compileall isolated_mamba_fusion"
"$PYTHON_BIN" -m compileall isolated_mamba_fusion
echo "[OK] Syntax check passed"

echo "------------------------------------------------------------"
echo "[INFO] Import check"
"$PYTHON_BIN" - <<'PY'
from isolated_mamba_fusion.modules import BaseFusion, DiffMambaFusion, ExistGuidedMambaFusion

print("Imported:", BaseFusion.__name__, DiffMambaFusion.__name__, ExistGuidedMambaFusion.__name__)
PY
echo "[OK] Import check passed"

echo "------------------------------------------------------------"
echo "[INFO] Running test_shapes.py"
"$PYTHON_BIN" isolated_mamba_fusion/tests/test_shapes.py
echo "[OK] test_shapes.py passed"

echo "------------------------------------------------------------"
echo "[INFO] Running test_backward_nan.py"
"$PYTHON_BIN" isolated_mamba_fusion/tests/test_backward_nan.py
echo "[OK] test_backward_nan.py passed"

echo "------------------------------------------------------------"
echo "[INFO] Running demo_random_forward.py"
"$PYTHON_BIN" isolated_mamba_fusion/demo/demo_random_forward.py
echo "[OK] demo_random_forward.py passed"

if [[ -f "isolated_mamba_fusion/tests/test_comprehensive.py" ]]; then
  echo "------------------------------------------------------------"
  echo "[INFO] Running test_comprehensive.py"
  "$PYTHON_BIN" isolated_mamba_fusion/tests/test_comprehensive.py
  echo "[OK] test_comprehensive.py passed"
else
  echo "------------------------------------------------------------"
  echo "[INFO] test_comprehensive.py not found; skipped"
fi

echo "============================================================"
echo "[OK] All isolated_mamba_fusion checks passed"
echo "============================================================"

