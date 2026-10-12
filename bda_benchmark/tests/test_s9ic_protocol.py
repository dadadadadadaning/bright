#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
train = (
    ROOT
    / "script/standard_ML/train_S9Ic_G1.py"
).read_text(encoding="utf-8")
geometry = (
    ROOT
    / "script/standard_ML/s9ic_g1_geometry.py"
).read_text(encoding="utf-8")
run = (
    ROOT
    / "run/run_S9Ic_G1_train.sh"
).read_text(encoding="utf-8")

required = [
    'EXPECTED_FEATURE_KEY',
    'build_unique_gaussian_targets',
    'reliable_geometry_loss',
    'checkpoint_step_',
    '"boundary_head_trained": True',
    '"boundary_permission_enabled_in_g1": False',
    '"rollback_enabled_in_g1": False',
    '"final_selection_locked": False',
]
for token in required:
    assert token in train, token

for token in (
    "peak_threshold: float = 0.35",
    "nms_kernel: int = 3",
    "min_peak_distance_native: int = 3",
    "max_markers_per_region: int = 4",
    "boundary_permission_enabled: bool = False",
    "rollback_enabled: bool = False",
):
    assert token in geometry, token

assert "--optimizer_steps 3000" in run
assert "--eval_every 1000" in run
assert "Test             : NOT ACCESSED" in run
print("S9-Ic-G1 locked protocol static test: PASS")
