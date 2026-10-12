#!/usr/bin/env python3
"""Compatibility launcher for S9-Ic-G2 (watershed + boundary gate)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.script.standard_ML.train_S9_instance_geometry import main


if __name__ == "__main__":
    raise SystemExit(main(["--variant", "g2", *sys.argv[1:]]))
