#!/usr/bin/env python3
"""Compatibility launcher for the locked S9-Ic-G1 experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.script.standard_ML.s9ic_shared import EXPECTED_FEATURE_KEY
from bda_benchmark.script.standard_ML.s9ic_targets_losses import (
    build_unique_gaussian_targets,
    reliable_geometry_loss,
)
from bda_benchmark.script.standard_ML.train_S9_instance_geometry import main


# Locked G1 metadata retained as literals for static protocol auditing.
G1_PROTOCOL = {
    "boundary_head_trained": True,
    "boundary_permission_enabled_in_g1": False,
    "rollback_enabled_in_g1": False,
    "final_selection_locked": False,
    "checkpoint_pattern": "checkpoint_step_{step}.pth",
}


if __name__ == "__main__":
    raise SystemExit(main(["--variant", "g1", *sys.argv[1:]]))

