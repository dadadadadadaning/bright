"""Compatibility entry for center-mechanism diagnostics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model.instance_geometry.metrics import (
    CenterMetricAccumulator,
    evaluate_center_mechanism,
)

__all__ = ["CenterMetricAccumulator", "evaluate_center_mechanism"]
