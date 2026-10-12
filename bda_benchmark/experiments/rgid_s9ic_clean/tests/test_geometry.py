from __future__ import annotations

import numpy as np

from rgid.config import RGIDConfig
from rgid.geometry import (
    RGIDDecoder,
    boundary_cut_score,
    extract_native_peaks,
    validate_geometry,
)
from rgid.targets_losses import binary_mask_to_instances
from rgid.metrics import aggregate_metric_dicts


def _parent_labels():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    return binary_mask_to_instances(mask)


def test_wide_flat_peak_produces_one_marker():
    center = np.zeros((8, 8), dtype=np.float32)
    center[3:5, 3:5] = 0.9
    peaks, raw = extract_native_peaks(center, _parent_labels(), RGIDConfig())
    assert raw == 4
    assert len(peaks) == 1


def test_two_far_peaks_are_retained():
    center = np.zeros((8, 8), dtype=np.float32)
    center[2, 2] = 0.9
    center[6, 6] = 0.8
    peaks, _ = extract_native_peaks(center, _parent_labels(), RGIDConfig())
    assert len(peaks) == 2


def test_two_near_peaks_keep_only_stronger():
    center = np.zeros((8, 8), dtype=np.float32)
    center[3, 3] = 0.9
    center[3, 5] = 0.8
    peaks, _ = extract_native_peaks(center, _parent_labels(), RGIDConfig())
    assert len(peaks) == 1
    assert peaks[0].score == np.float32(0.9)


def test_strong_boundary_cut_is_high_and_weak_is_zero():
    parent = np.zeros((20, 20), dtype=bool)
    parent[2:18, 2:18] = True
    children = np.zeros((20, 20), dtype=np.int32)
    children[2:18, 2:10] = 1
    children[2:18, 10:18] = 2
    strong = np.zeros((20, 20), dtype=np.float32)
    # Cover the dilated three-pixel cut band, not only its center line.
    strong[:, 8:12] = 1.0
    weak = np.zeros_like(strong)
    assert boundary_cut_score(children, parent, strong, band_radius=1) > 0.5
    assert boundary_cut_score(children, parent, weak, band_radius=1) == 0.0


def test_tiny_child_triggers_rollback():
    parent = np.zeros((20, 20), dtype=bool)
    parent[2:18, 2:18] = True
    children = np.zeros((20, 20), dtype=np.int32)
    children[parent] = 1
    children[2:4, 2:4] = 2
    reasons = validate_geometry(
        children,
        parent,
        [(10, 10), (2, 2)],
        RGIDConfig(min_child_area=32, min_child_parent_ratio=0.05),
    )
    assert "child_too_small" in reasons
    assert "child_ratio_too_small" in reasons


def test_g2_without_boundary_rejects_split_and_restores_parent():
    parent = _parent_labels() > 0
    center = np.zeros((8, 8), dtype=np.float32)
    center[2, 2] = 0.99
    center[5, 5] = 0.95
    boundary = np.zeros((8, 8), dtype=np.float32)
    result = RGIDDecoder(RGIDConfig()).decode(
        center,
        boundary,
        parent,
        mode="g2",
        inputs_are_logits=False,
    )
    assert int(result.instance_labels.max()) == 1
    assert result.stats.boundary_gate_rejected_regions == 1
    assert result.stats.split_accepted_regions == 0


def test_g1_splits_same_parent_without_boundary_gate():
    parent = _parent_labels() > 0
    center = np.zeros((8, 8), dtype=np.float32)
    center[2, 2] = 0.99
    center[5, 5] = 0.95
    boundary = np.zeros((8, 8), dtype=np.float32)
    result = RGIDDecoder(RGIDConfig()).decode(
        center,
        boundary,
        parent,
        mode="g1",
        inputs_are_logits=False,
    )
    assert int(result.instance_labels.max()) == 2
    assert result.stats.split_accepted_regions == 1


def test_global_metric_aggregation_sums_counts_before_f1():
    rows = [
        {
            "gt_instances": 1,
            "pred_instances": 1,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 0,
            "matched_iou_sum": 0.8,
            "merge": 0,
            "split": 0,
        },
        {
            "gt_instances": 9,
            "pred_instances": 9,
            "true_positive": 0,
            "false_positive": 9,
            "false_negative": 9,
            "matched_iou_sum": 0.0,
            "merge": 2,
            "split": 3,
        },
    ]
    aggregate = aggregate_metric_dicts(rows)
    assert aggregate["true_positive"] == 1
    assert aggregate["false_positive"] == 9
    assert aggregate["false_negative"] == 9
    assert abs(aggregate["precision"] - 0.1) < 1e-12
    assert abs(aggregate["recall"] - 0.1) < 1e-12
    assert abs(aggregate["f1"] - 0.1) < 1e-12
    assert abs(aggregate["mean_matched_iou"] - 0.8) < 1e-12
    assert aggregate["merge"] == 2
    assert aggregate["split"] == 3
