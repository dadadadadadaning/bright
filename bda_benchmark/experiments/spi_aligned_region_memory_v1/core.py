from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    make_region_ids,
    split_regions_by_building,
)


PROTOCOL = "SPI_ALIGNED_REGION_MEMORY_V1"
REGION_VARIANTS = ("r1_grid", "r2_slico", "r3_shuffled", "r4_learned")
KEY_GROUPS = ("rgb", "sar", "relation_aligned", "multiring")
FORBIDDEN_KEY_GROUPS = ("p3_state", "quality", "geometry", "label", "event")


@dataclass(frozen=True)
class Admission:
    region_gain_pp: float = 0.10
    memory_gain_pp: float = 0.20
    final_gain_pp: float = 0.30
    minimum_nonnegative_events: int = 10


class SoftRegionTokenizer(nn.Module):
    """Small differentiable SSN-style partitioner driven by pre-event RGB.

    It learns an embedding, compares every pixel with a regular grid of seed
    embeddings, and returns soft assignments.  The module does not predict a
    damage class and never receives post-event SAR or P3 class probabilities.
    """

    def __init__(self, embedding_dim: int = 24, seed_rows: int = 20, seed_cols: int = 20) -> None:
        super().__init__()
        self.seed_rows = int(seed_rows)
        self.seed_cols = int(seed_cols)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, bias=False),
            nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv2d(24, embedding_dim, 3, padding=1, bias=False),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.10)))
        self.log_spatial_weight = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, rgb: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        feature = F.interpolate(self.encoder(rgb.float()), size=size, mode="bilinear", align_corners=False)
        feature = F.normalize(feature, dim=1)
        seed = F.adaptive_avg_pool2d(feature, (self.seed_rows, self.seed_cols))
        seed = F.normalize(seed.flatten(2), dim=1)
        similarity = torch.einsum("bchw,bck->bkhw", feature, seed)

        yy = torch.linspace(0.0, 1.0, size[0], device=rgb.device, dtype=feature.dtype)
        xx = torch.linspace(0.0, 1.0, size[1], device=rgb.device, dtype=feature.dtype)
        sy = torch.linspace(0.0, 1.0, self.seed_rows, device=rgb.device, dtype=feature.dtype)
        sx = torch.linspace(0.0, 1.0, self.seed_cols, device=rgb.device, dtype=feature.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        sgy, sgx = torch.meshgrid(sy, sx, indexing="ij")
        distance = (gy[None] - sgy.flatten()[:, None, None]).square()
        distance = distance + (gx[None] - sgx.flatten()[:, None, None]).square()
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        spatial = self.log_spatial_weight.exp().clamp(0.05, 20.0)
        assignment = torch.softmax((similarity - spatial * distance[None]) / temperature, dim=1)
        return assignment, feature


def learned_region_loss(
    assignment: torch.Tensor,
    feature: torch.Tensor,
    building: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruction + compactness + building-boundary preservation.

    GT building masks are permitted only here on Train.  They supervise the
    partitioner but are not stored in retrieval Keys and are not needed at
    inference.
    """
    b, k, h, w = assignment.shape
    flat_a = assignment.flatten(2)
    mass = flat_a.sum(2).clamp_min(1e-6)
    flat_f = feature.flatten(2)
    seed_f = torch.einsum("bkn,bcn->bkc", flat_a, flat_f) / mass[:, :, None]
    reconstructed = torch.einsum("bkn,bkc->bcn", flat_a, seed_f).view_as(feature)
    reconstruction = (reconstructed - feature).square().mean()

    target = F.interpolate(building.float(), (h, w), mode="nearest")
    if target.ndim == 3:
        target = target[:, None]
    target = (target > 0).float()
    seed_b = torch.einsum("bkn,bcn->bkc", flat_a, target.flatten(2)) / mass[:, :, None]
    reconstructed_b = torch.einsum("bkn,bkc->bcn", flat_a, seed_b).view(b, 1, h, w)
    boundary = F.binary_cross_entropy(reconstructed_b.clamp(1e-5, 1 - 1e-5), target)
    entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(1).mean() / math.log(float(k))
    balance = (mass / mass.sum(1, keepdim=True).clamp_min(1.0) - 1.0 / k).square().mean()
    total = reconstruction + 0.50 * boundary + 0.02 * entropy + 0.10 * balance
    return total, {"reconstruction": reconstruction, "boundary": boundary, "entropy": entropy, "balance": balance}


def region_ids(
    pre_rgb: torch.Tensor,
    p_building: torch.Tensor,
    size: tuple[int, int],
    variant: str,
    segments: int,
    compactness: float,
    learned: SoftRegionTokenizer | None = None,
) -> torch.Tensor:
    if variant not in REGION_VARIANTS:
        raise ValueError(f"unknown region variant: {variant}")
    modes = {"r1_grid": "grid", "r2_slico": "slico", "r3_shuffled": "shuffled_slico"}
    if variant == "r4_learned":
        if learned is None:
            raise RuntimeError("r4_learned requires a frozen tokenizer checkpoint")
        assignment, _ = learned(pre_rgb, size)
        ids = assignment.argmax(1)
    else:
        ids = make_region_ids(pre_rgb, size, mode=modes[variant], n_segments=segments, compactness=compactness)
    return split_regions_by_building(ids, p_building)


def audit_contract(metadata: Mapping[str, object], split: str) -> None:
    key_groups = tuple(metadata.get("key_groups", ()))
    forbidden = sorted(set(key_groups) & set(FORBIDDEN_KEY_GROUPS))
    if forbidden:
        raise RuntimeError(f"retrieval Key contains forbidden shortcut fields: {forbidden}")
    if split != "train" and bool(metadata.get("memory_write_allowed", False)):
        raise RuntimeError("Validation/Test records must be read-only for Memory")
    if bool(metadata.get("majority_vote", False)):
        raise RuntimeError("CSSAD-style hard majority vote is forbidden in this protocol")


__all__ = [
    "Admission", "KEY_GROUPS", "PROTOCOL", "REGION_VARIANTS", "SoftRegionTokenizer",
    "audit_contract", "learned_region_loss", "region_ids",
]

