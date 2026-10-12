from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryCalibration:
    boundary_positive_median: float
    boundary_interior_median: float
    boundary_gate_threshold: float
    num_boundary_pixels: int
    num_interior_pixels: int


class BoundaryThresholdCalibrator:
    def __init__(self) -> None:
        self._positive: List[np.ndarray] = []
        self._interior: List[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        boundary_logits,
        boundary_target,
        building_mask=None,
    ) -> None:
        logits = torch.as_tensor(boundary_logits).float()
        target = torch.as_tensor(boundary_target).float()
        if logits.ndim == 3:
            logits = logits[:, None]
        if target.ndim == 3:
            target = target[:, None]
        if logits.shape != target.shape:
            raise ValueError("boundary logits and target must have the same shape")
        probability = torch.sigmoid(logits)
        if building_mask is None:
            building = torch.ones_like(target, dtype=torch.bool)
        else:
            building = torch.as_tensor(building_mask, device=target.device).bool()
            if building.ndim == 3:
                building = building[:, None]
            if building.shape[-2:] != target.shape[-2:]:
                building = F.interpolate(
                    building.float(),
                    size=target.shape[-2:],
                    mode="nearest",
                ).bool()
            if building.shape[0] != target.shape[0]:
                raise ValueError("building_mask batch size differs from target")
        positive = target >= 0.5
        interior = building & ~positive
        if positive.any():
            self._positive.append(
                probability[positive].detach().cpu().numpy().astype(np.float32)
            )
        if interior.any():
            self._interior.append(
                probability[interior].detach().cpu().numpy().astype(np.float32)
            )

    def compute(self) -> BoundaryCalibration:
        if not self._positive:
            raise RuntimeError("no positive boundary pixels were collected")
        if not self._interior:
            raise RuntimeError("no building-interior pixels were collected")
        positive = np.concatenate(self._positive)
        interior = np.concatenate(self._interior)
        positive_median = float(np.median(positive))
        interior_median = float(np.median(interior))
        return BoundaryCalibration(
            boundary_positive_median=positive_median,
            boundary_interior_median=interior_median,
            boundary_gate_threshold=(positive_median + interior_median) / 2.0,
            num_boundary_pixels=int(positive.size),
            num_interior_pixels=int(interior.size),
        )
