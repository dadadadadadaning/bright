from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


@dataclass(frozen=True)
class SplitRiskConfig:
    """Train-only calibrated policy for deciding whether one LocCC is split.

    The defaults are deliberately diagnostic rather than paper constants.  A
    final threshold must be selected on a train-only calibration partition and
    recorded in RGID provenance before validation is evaluated.
    """

    score_threshold: float = 0.55
    area_reference: int = 4096
    peak_threshold: float = 0.35
    peak_nms_kernel: int = 3
    minimum_stable_peaks: int = 2
    weight_area: float = 0.15
    weight_elongation: float = 0.10
    weight_holes: float = 0.05
    weight_entropy: float = 0.20
    weight_peaks: float = 0.30
    weight_boundary: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0,1]")
        if self.area_reference <= 0:
            raise ValueError("area_reference must be positive")
        if self.peak_nms_kernel <= 0 or self.peak_nms_kernel % 2 == 0:
            raise ValueError("peak_nms_kernel must be a positive odd integer")
        weights = (
            self.weight_area,
            self.weight_elongation,
            self.weight_holes,
            self.weight_entropy,
            self.weight_peaks,
            self.weight_boundary,
        )
        if any(value < 0 for value in weights):
            raise ValueError("split-risk weights must be nonnegative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("split-risk weights must sum to one")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SplitRiskEvidence:
    parent_id: int
    area: int
    area_score: float
    elongation_score: float
    holes_score: float
    entropy_score: float
    peak_count: int
    peak_score: float
    boundary_score: float
    risk_score: float
    child_count: int
    accepted: bool
    rejection_reason: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def _as_bool_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    output = np.asarray(torch.as_tensor(mask).detach().cpu(), dtype=bool)
    if output.ndim != 2 or not output.any():
        raise ValueError("parent mask must be a non-empty [H,W] array")
    return output


def _mean_entropy(probabilities: torch.Tensor, mask: np.ndarray) -> float:
    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("probabilities must be [4,H,W]")
    if tuple(probabilities.shape[-2:]) != tuple(mask.shape):
        raise ValueError("probabilities and parent mask have different shapes")
    selected = probabilities.float()[:, torch.from_numpy(mask).to(probabilities.device)]
    selected = selected.clamp_min(1e-8)
    entropy = -(selected * selected.log()).sum(dim=0) / np.log(4.0)
    return float(entropy.mean().clamp(0.0, 1.0).item())


def _count_center_peaks(
    center_probability: torch.Tensor,
    mask: np.ndarray,
    config: SplitRiskConfig,
) -> int:
    center = torch.as_tensor(center_probability).detach().float().cpu()
    if center.ndim == 3 and center.shape[0] == 1:
        center = center[0]
    if center.ndim != 2:
        raise ValueError("center_probability must be [Hh,Wh] or [1,Hh,Wh]")
    parent = torch.from_numpy(mask.astype(np.float32))[None, None]
    parent = F.interpolate(parent, size=center.shape, mode="nearest")[0, 0].bool()
    pooled = F.max_pool2d(
        center[None, None],
        kernel_size=config.peak_nms_kernel,
        stride=1,
        padding=config.peak_nms_kernel // 2,
    )[0, 0]
    peaks = parent & (center >= config.peak_threshold) & (center == pooled)
    labels, count = ndimage.label(peaks.numpy(), structure=np.ones((3, 3)))
    del labels
    return int(count)


def score_split_risk(
    *,
    parent_id: int,
    parent_mask: np.ndarray | torch.Tensor,
    probabilities: torch.Tensor,
    center_probability: torch.Tensor,
    boundary_probability: torch.Tensor,
    child_count: int,
    config: SplitRiskConfig,
) -> SplitRiskEvidence:
    """Compute prediction-only evidence for split-on-demand.

    No semantic or instance ground truth is accepted by this function.  That
    makes it safe to run at validation/inference after the policy is frozen.
    """

    mask = _as_bool_mask(parent_mask)
    area = int(mask.sum())
    area_score = float(np.clip(np.log1p(area) / np.log1p(config.area_reference), 0, 1))

    yy, xx = np.nonzero(mask)
    height = int(yy.max() - yy.min() + 1)
    width = int(xx.max() - xx.min() + 1)
    aspect = max(height, width) / max(1, min(height, width))
    elongation_score = float(np.clip((aspect - 1.0) / 3.0, 0, 1))

    filled = ndimage.binary_fill_holes(mask)
    hole_pixels = int((filled & ~mask).sum())
    holes_score = float(np.clip(hole_pixels / max(1, area), 0, 1))
    entropy_score = _mean_entropy(probabilities, mask)
    peak_count = _count_center_peaks(center_probability, mask, config)
    peak_score = float(np.clip((peak_count - 1) / 3.0, 0, 1))

    boundary = torch.as_tensor(boundary_probability).detach().float().cpu()
    if boundary.ndim == 3 and boundary.shape[0] == 1:
        boundary = boundary[0]
    if boundary.ndim != 2:
        raise ValueError("boundary_probability must be [Hh,Wh] or [1,Hh,Wh]")
    boundary = F.interpolate(
        boundary[None, None], size=mask.shape, mode="bilinear", align_corners=False
    )[0, 0]
    boundary_score = float(boundary[torch.from_numpy(mask)].mean().clamp(0, 1).item())

    score = float(
        config.weight_area * area_score
        + config.weight_elongation * elongation_score
        + config.weight_holes * holes_score
        + config.weight_entropy * entropy_score
        + config.weight_peaks * peak_score
        + config.weight_boundary * boundary_score
    )
    if int(child_count) < 2:
        accepted, reason = False, "rgid_has_fewer_than_two_children"
    elif peak_count < config.minimum_stable_peaks:
        accepted, reason = False, "insufficient_stable_centers"
    elif score < config.score_threshold:
        accepted, reason = False, "risk_below_threshold"
    else:
        accepted, reason = True, "accepted"
    return SplitRiskEvidence(
        parent_id=int(parent_id),
        area=area,
        area_score=area_score,
        elongation_score=elongation_score,
        holes_score=holes_score,
        entropy_score=entropy_score,
        peak_count=peak_count,
        peak_score=peak_score,
        boundary_score=boundary_score,
        risk_score=score,
        child_count=int(child_count),
        accepted=accepted,
        rejection_reason=reason,
    )


def selective_partition(
    *,
    parent_labels: np.ndarray | torch.Tensor,
    rgid_child_labels: np.ndarray | torch.Tensor,
    evidence_by_parent: Mapping[int, SplitRiskEvidence],
) -> np.ndarray:
    """Use RGID children only for accepted parents while preserving support."""

    parents = np.asarray(torch.as_tensor(parent_labels).detach().cpu(), dtype=np.int32)
    children = np.asarray(
        torch.as_tensor(rgid_child_labels).detach().cpu(), dtype=np.int32
    )
    if parents.ndim != 2 or children.shape != parents.shape:
        raise ValueError("parent and child labels must be matching [H,W] arrays")
    if not np.array_equal(parents > 0, children > 0):
        raise ValueError("RGID and LocCC must cover exactly the same support")

    output = np.zeros_like(parents, dtype=np.int32)
    next_id = 1
    for parent_id in sorted(int(value) for value in np.unique(parents) if value > 0):
        parent = parents == parent_id
        evidence = evidence_by_parent.get(parent_id)
        use_children = bool(evidence is not None and evidence.accepted)
        ids: Sequence[int] = (
            [int(value) for value in np.unique(children[parent]) if value > 0]
            if use_children
            else []
        )
        if len(ids) < 2:
            output[parent] = next_id
            next_id += 1
            continue
        for child_id in ids:
            child = parent & (children == child_id)
            if child.any():
                output[child] = next_id
                next_id += 1
    if not np.array_equal(output > 0, parents > 0):
        raise RuntimeError("selective partition did not conserve parent support")
    return output


def random_matched_partition(
    *,
    parent_labels: np.ndarray | torch.Tensor,
    rgid_child_labels: np.ndarray | torch.Tensor,
    split_parent_count: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """D3 control: randomly split the same number of splittable LocCC parents."""

    parents = np.asarray(torch.as_tensor(parent_labels).detach().cpu(), dtype=np.int32)
    children = np.asarray(
        torch.as_tensor(rgid_child_labels).detach().cpu(), dtype=np.int32
    )
    if parents.ndim != 2 or children.shape != parents.shape:
        raise ValueError("parent and child labels must be matching [H,W] arrays")
    if not np.array_equal(parents > 0, children > 0):
        raise ValueError("RGID and LocCC must cover exactly the same support")
    candidates = []
    for parent_id in sorted(int(value) for value in np.unique(parents) if value > 0):
        child_count = len(np.unique(children[parents == parent_id]))
        if child_count >= 2:
            candidates.append(parent_id)
    if split_parent_count < 0 or split_parent_count > len(candidates):
        raise ValueError("split_parent_count exceeds splittable parents")
    generator = np.random.default_rng(int(seed))
    selected = sorted(
        int(value)
        for value in generator.choice(candidates, size=split_parent_count, replace=False)
    )
    selected_set = set(selected)
    output = np.zeros_like(parents, dtype=np.int32)
    next_id = 1
    for parent_id in sorted(int(value) for value in np.unique(parents) if value > 0):
        parent = parents == parent_id
        if parent_id not in selected_set:
            output[parent] = next_id
            next_id += 1
            continue
        for child_id in sorted(int(value) for value in np.unique(children[parent])):
            child = parent & (children == child_id)
            output[child] = next_id
            next_id += 1
    return output, selected


__all__ = [
    "SplitRiskConfig",
    "SplitRiskEvidence",
    "score_split_risk",
    "selective_partition",
    "random_matched_partition",
]
