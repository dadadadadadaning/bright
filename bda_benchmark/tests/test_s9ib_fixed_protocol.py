#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static protocol test for the S9-Ib evaluator."""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "script/standard_ML/eval_S9Ib_conservative_center.py"
source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)

required_literals = {
    "center_threshold=0.50",
    "min_region_area_to_split=512",
    "min_center_area=4",
    "min_center_distance=10",
    "max_markers_per_region=4",
    '"PROCEED_S9B_CONSERVATIVE_FEEDBACK"',
    '"PROCEED_S9IC_RELIABLE_GEOMETRY"',
}

for literal in required_literals:
    assert literal in source, f"Missing fixed protocol literal: {literal}"

for forbidden in (
    "--center_threshold",
    "--min_region_area_to_split",
    "--min_center_area",
    "--min_center_distance",
    "--max_markers_per_region",
):
    assert forbidden not in source, f"Forbidden tuning CLI exposed: {forbidden}"

assert tree is not None
print("S9-Ib fixed protocol static test: PASS")
