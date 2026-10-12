"""Exact existing-weight A1/region paths and paired Test accounting."""
from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
    feature_branches,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import (
    close_report,
    transition_counts,
    transition_summary,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import confusion_update


ARMS = (
    "a1_pixel", "pixel_grid", "feature_mean", "region_mean", "grid_probq", "region",
)
PAIRS = (
    "pixel_grid__grid_probq", "pixel_grid__feature_mean", "feature_mean__region_mean",
    "region_mean__region", "a1_pixel__grid_probq",
)
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")


def read_test_names(path, train_names, val_names, expected_count=878):
    """Read the canonical Test split and reject any Train/Val or unsafe entry."""
    path = Path(path)
    if path.name != "test_set.txt":
        raise ValueError("Only the standard test_set.txt is accepted")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(expected_count, int) or expected_count < 1 or len(names) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} Test image names, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("Test list contains duplicate names")
    if any(SAFE_NAME.fullmatch(name) is None for name in names):
        raise ValueError("Test list contains an invalid or unsafe sample identifier")
    if set(names) & (set(train_names) | set(val_names)):
        raise ValueError("Test list overlaps the Train or Val split")
    return names


def sample_paths(dataset, name):
    """Paths used by the established labelled BRIGHT data loader."""
    if not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None:
        raise ValueError("Invalid sample identifier")
    root = Path(dataset)
    return {
        "pre": root / "pre-event" / f"{name}_pre_disaster.tif",
        "post": root / "post-event" / f"{name}_post_disaster.tif",
        "label": root / "target" / f"{name}_building_damage.tif",
    }


@torch.no_grad()
def evaluate_image(front, result, feature, output_size):
    """Use unchanged F/Q/H and the original region output from one front call."""
    if not isinstance(result, dict) or any(
        key not in result for key in ("q", "ids", "count", "shape", "old_pixel", "prob")
    ):
        raise ValueError("Incomplete front result")
    if tuple(feature.shape[-2:]) != tuple(result["shape"]):
        raise ValueError("A1 feature and Q must share one grid")
    if tuple(result["old_pixel"].shape[-2:]) != tuple(output_size):
        raise ValueError("Native A1 output and requested image size disagree")
    computed, checks, (direct_logits, _) = feature_branches(
        feature.float(), result["q"], result["ids"], result["count"],
        front.base.clf_2, output_size,
    )
    checks["a1_native_replay"] = close_report(
        F.interpolate(direct_logits.float(), size=output_size, mode="bilinear", align_corners=False).softmax(1),
        result["old_pixel"].float().softmax(1),
    )
    probabilities = {
        "a1_pixel": result["old_pixel"].float().softmax(1),
        **computed,
        "region": result["prob"].float(),
    }
    if set(probabilities) != set(ARMS):
        raise RuntimeError("Unexpected Test prediction paths")
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        probability_prediction,
    )
    for name in ARMS:
        probability_prediction(probabilities[name], output_size)
    return probabilities, checks


class Accumulator:
    """Per-image, per-event and paired pixel counts for a single Test pass."""

    def __init__(self, arms=ARMS, pairs=PAIRS):
        if len(set(arms)) != len(arms) or not arms:
            raise ValueError("Prediction paths must be unique and nonempty")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Comparison pairs must be unique")
        for pair in pairs:
            names = pair.split("__")
            if len(names) != 2 or any(name not in arms for name in names):
                raise ValueError("Comparison pair refers to a missing prediction path")
        self.arms = tuple(arms)
        self.pairs = tuple(pairs)
        self.image_count = 0
        self.cms = {name: torch.zeros(4, 4, dtype=torch.long) for name in self.arms}
        self.per_event = {}
        self.joints = {name: torch.zeros(4, 4, 4, dtype=torch.long) for name in self.pairs}

    def add(self, name, event, target, predictions):
        if not isinstance(name, str) or not name or not isinstance(event, str) or not event:
            raise ValueError("Each Test image needs a name and event")
        if set(predictions) != set(self.arms):
            raise ValueError("Prediction paths differ from the fixed evaluation plan")
        y = target.detach().cpu().long()
        if y.ndim != 3 or y.shape[0] != 1:
            raise ValueError("Expected one [1,H,W] label image")
        converted = {}
        for arm in self.arms:
            p = predictions[arm]
            if p.shape != y.shape or p.dtype not in (torch.int32, torch.int64, torch.uint8):
                raise ValueError("Each prediction must be an integer [1,H,W] label map")
            p = p.detach().cpu().long()
            if bool(((p < 0) | (p >= 4)).any()):
                raise ValueError("Prediction class outside [0,3]")
            converted[arm] = p
        # Compute first, then commit: an invalid later arm cannot leave partial
        # global or per-event counts behind.
        rows = {}
        computed = {}
        for arm in self.arms:
            cm = torch.zeros(4, 4, dtype=torch.long)
            confusion_update(cm, y, converted[arm])
            computed[arm] = cm
            rows[arm] = cm.tolist()
        joints = {}
        for pair in self.pairs:
            old, new = pair.split("__")
            joints[pair] = transition_counts(y, converted[old], converted[new])
        event_cms = self.per_event.setdefault(event, {
            arm: torch.zeros(4, 4, dtype=torch.long) for arm in self.arms
        })
        for arm in self.arms:
            self.cms[arm] += computed[arm]
            event_cms[arm] += computed[arm]
        for pair in self.pairs:
            self.joints[pair] += joints[pair]
        self.image_count += 1
        return rows

    def finish(self):
        return {
            "image_count": self.image_count,
            "metrics": {arm: metrics(cm) for arm, cm in self.cms.items()},
            "per_event": {
                event: {arm: metrics(cm) for arm, cm in values.items()}
                for event, values in self.per_event.items()
            },
            "transitions": {pair: transition_summary(joint) for pair, joint in self.joints.items()},
        }
