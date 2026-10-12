#!/usr/bin/env python3
"""Synthetic instance P/R/F1, IoU, merge, and split aggregation test."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.script.standard_ML.s9_instance_common import (
    InstanceMetricAccumulator,
    evaluate_instance_geometry,
    instances_from_bag_specs,
    instances_from_masks,
)


@dataclass
class Spec:
    batch_index: int
    mask: torch.Tensor
    label: int = 1


left = np.zeros((32, 32), dtype=np.bool_)
right = np.zeros_like(left)
left[4:14, 4:14] = True
right[4:14, 14:24] = True
merged = left | right
targets = instances_from_bag_specs(
    [Spec(0, torch.from_numpy(left)), Spec(0, torch.from_numpy(right))]
)
predicted = instances_from_masks([merged], batch_index=0)
merge_report = evaluate_instance_geometry(predicted, targets)
assert merge_report["matched_instances"] == 1
assert merge_report["merge_pred_instances"] == 1

whole = np.zeros((32, 32), dtype=np.bool_)
whole[18:28, 4:24] = True
first_half = np.zeros_like(whole)
second_half = np.zeros_like(whole)
first_half[18:28, 4:14] = True
second_half[18:28, 14:24] = True
targets = instances_from_bag_specs([Spec(0, torch.from_numpy(whole))])
predicted = instances_from_masks([first_half, second_half], batch_index=0)
split_report = evaluate_instance_geometry(predicted, targets)
assert split_report["matched_instances"] == 1
assert split_report["split_gt_instances"] == 1

accumulator = InstanceMetricAccumulator()
accumulator.update(merge_report)
accumulator.update(split_report)
result = accumulator.compute()
assert result["target_instances"] == 3
assert result["predicted_instances"] == 3
assert result["matched_instances"] == 2
assert np.isclose(result["precision"], 2.0 / 3.0)
assert np.isclose(result["recall"], 2.0 / 3.0)
assert result["merge"] == 1
assert result["split"] == 1
assert result["mean_matched_iou"] >= 0.5
print("S9 instance geometry metrics test: PASS")
