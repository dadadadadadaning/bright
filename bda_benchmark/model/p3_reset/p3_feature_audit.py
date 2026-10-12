from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..DamageFormer_S8_MICB_DSMIL import connected_components_4


@dataclass(frozen=True)
class RegionReadoutBatch:
    features: torch.Tensor
    labels: torch.Tensor
    areas: torch.Tensor
    batch_indices: torch.Tensor


def _resize_mask(mask: torch.Tensor, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return F.interpolate(mask[None, None].float().to(device), size=size, mode="nearest")[0, 0]


def pool_semantic_regions(
    feature: torch.Tensor,
    labels_loc: torch.Tensor,
    labels_damage: torch.Tensor,
    min_area: int = 32,
    minimum_purity: float = 0.75,
) -> RegionReadoutBatch:
    """Oracle-only region pooling for A0/B0--B2.

    Regions come only from the binary GT building support. Damage labels assign
    a majority target *after* region construction; they never determine region
    boundaries. This avoids leaking the I/D/X answer into the input geometry.
    """

    vectors: List[torch.Tensor] = []
    labels: List[int] = []
    areas: List[int] = []
    batch_indices: List[int] = []
    loc = labels_loc[:, 0] if labels_loc.ndim == 4 else labels_loc
    damage = labels_damage[:, 0] if labels_damage.ndim == 4 else labels_damage
    if loc.ndim != 3 or damage.shape != loc.shape:
        raise ValueError("localisation and damage targets must share [B,H,W]")
    if not 0.0 < minimum_purity <= 1.0:
        raise ValueError("minimum_purity must lie in (0,1]")
    for batch_index in range(loc.shape[0]):
        building = (loc[batch_index] == 1).detach().cpu()
        for component in connected_components_4(building):
            component_tensor = torch.from_numpy(component).to(damage.device)
            area = int(component_tensor.sum())
            if area < min_area:
                continue
            values = damage[batch_index][component_tensor]
            counts = torch.stack([(values == label).sum() for label in (1, 2, 3)])
            majority_count, majority_index = counts.max(0)
            if float(majority_count) / max(area, 1) < minimum_purity:
                continue
            mask = _resize_mask(component_tensor, feature.shape[-2:], feature.device)
            denominator = mask.sum()
            if float(denominator) < 1.0:
                continue
            vectors.append((feature[batch_index] * mask).flatten(1).sum(1) / denominator)
            labels.append(int(majority_index))
            areas.append(area)
            batch_indices.append(batch_index)
    if not vectors:
        return RegionReadoutBatch(
            features=feature.new_zeros((0, feature.shape[1])),
            labels=torch.empty(0, dtype=torch.long, device=feature.device),
            areas=torch.empty(0, dtype=torch.long, device=feature.device),
            batch_indices=torch.empty(0, dtype=torch.long, device=feature.device),
        )
    return RegionReadoutBatch(
        features=torch.stack(vectors),
        labels=torch.tensor(labels, dtype=torch.long, device=feature.device),
        areas=torch.tensor(areas, dtype=torch.long, device=feature.device),
        batch_indices=torch.tensor(batch_indices, dtype=torch.long, device=feature.device),
    )


class FrozenFeatureReadout(nn.Module):
    """One normalisation plus one linear layer for a fair frozen-feature audit."""

    def __init__(self, in_channels: int, classes: int = 3) -> None:
        super().__init__()
        self.normalise = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.classifier = nn.Linear(in_channels, classes)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 2:
            raise ValueError(f"readout expects [regions,C], got {tuple(vectors.shape)}")
        return self.classifier(self.normalise(vectors))
