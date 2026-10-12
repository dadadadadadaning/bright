from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "SPI_BUILDING_RELATIONAL_MEMORY_V1"
VARIANTS = (
    "m0_rgb_mean",
    "m1_unaligned_multistat",
    "m2_aligned_multistat",
    "m3_aligned_multiring",
    "m4_aligned_relation",
    "m5_shuffled_relation",
    "m6_aligned_relation_quality",
)


@dataclass(frozen=True)
class VariantSpec:
    rgb: bool
    sar_group: str | None
    relation: str | None
    multiring: bool
    aggregation: str
    shuffled_geometry: bool = False
    quality_gate: bool = False


SPECS = {
    "m0_rgb_mean": VariantSpec(True, None, None, False, "mean"),
    "m1_unaligned_multistat": VariantSpec(True, "sar", "relation_unaligned", False, "multistat"),
    "m2_aligned_multistat": VariantSpec(True, "sar_aligned", "relation_aligned", False, "multistat"),
    "m3_aligned_multiring": VariantSpec(True, "sar_aligned", "relation_aligned", True, "multistat"),
    "m4_aligned_relation": VariantSpec(True, "sar_aligned", "relation_aligned", True, "relation"),
    "m5_shuffled_relation": VariantSpec(
        True, "sar_aligned", "relation_aligned", True, "relation", shuffled_geometry=True
    ),
    "m6_aligned_relation_quality": VariantSpec(
        True, "sar_aligned", "relation_aligned", True, "relation", quality_gate=True
    ),
}


class BranchEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(64, output_dim * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(1, keepdim=True).clamp_min(1).to(value.dtype)
    return (value * mask[:, :, None]).sum(1) / count


def _masked_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    minimum = torch.finfo(value.dtype).min
    return value.masked_fill(~mask[:, :, None], minimum).max(1).values


def _masked_quantile(value: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    rows = []
    for batch in range(value.shape[0]):
        valid = value[batch][mask[batch]]
        rows.append(torch.quantile(valid, q, dim=0) if valid.numel() else value.new_zeros(value.shape[-1]))
    return torch.stack(rows)


def _roll_valid_positions(position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Deterministic wrong-structure control; content and token counts stay unchanged."""
    output = position.clone()
    for batch in range(position.shape[0]):
        indices = torch.nonzero(mask[batch], as_tuple=False).flatten()
        if indices.numel() > 1:
            output[batch, indices] = position[batch, indices.roll(1)]
    return output


class LocalRelationLayer(nn.Module):
    """One low-capacity relation update using relative position and local kNN."""

    def __init__(self, dim: int, neighbours: int = 4) -> None:
        super().__init__()
        self.neighbours = int(neighbours)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.log_distance_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        token: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.query(token), self.key(token), self.value(token)
        score = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(float(token.shape[-1]))
        distance = torch.cdist(position.float(), position.float()).to(score.dtype)
        score = score - self.log_distance_scale.exp().clamp(0.01, 20.0) * distance
        valid_pair = mask[:, :, None] & mask[:, None, :]
        count = token.shape[1]
        take = min(max(self.neighbours, 1), count)
        nearest = distance.masked_fill(~valid_pair, float("inf")).topk(take, dim=-1, largest=False).indices
        local = torch.zeros_like(valid_pair)
        local.scatter_(2, nearest, True)
        valid_pair &= local
        score = score.masked_fill(~valid_pair, -1e4)
        if quality is not None:
            # Reliability changes how much a neighbour contributes, but is not
            # concatenated into the semantic retrieval content.
            score = score + quality.clamp_min(1e-4).log()[:, None, :]
        attention = score.softmax(-1) * mask[:, :, None].to(score.dtype)
        context = torch.einsum("bnm,bmd->bnd", attention, v)
        return self.norm(token + self.out(context)) * mask[:, :, None]


class BuildingRelationalEncoder(nn.Module):
    """Factorized node encoder plus controlled building-level aggregation."""

    def __init__(
        self,
        group_dims: Mapping[str, int],
        variant: str,
        node_dim: int = 128,
        key_dim: int = 128,
    ) -> None:
        super().__init__()
        if variant not in SPECS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.spec = SPECS[variant]
        self.rgb = BranchEncoder(group_dims["rgb"], 32) if self.spec.rgb else None
        self.sar = (
            BranchEncoder(group_dims[self.spec.sar_group], 32)
            if self.spec.sar_group else None
        )
        self.relation = (
            BranchEncoder(group_dims[self.spec.relation], 48) if self.spec.relation else None
        )
        self.multiring = BranchEncoder(group_dims["multiring"], 16) if self.spec.multiring else None
        branch_dim = sum((32 if self.rgb else 0, 32 if self.sar else 0, 48 if self.relation else 0, 16 if self.multiring else 0))
        self.node_projection = nn.Sequential(
            nn.LayerNorm(branch_dim), nn.Linear(branch_dim, node_dim), nn.GELU(), nn.LayerNorm(node_dim)
        )
        self.relation_layer = LocalRelationLayer(node_dim) if self.spec.aggregation == "relation" else None
        self.anomaly = nn.Linear(node_dim, 1)
        statistics = 1 if self.spec.aggregation == "mean" else 5
        self.building_projection = nn.Sequential(
            nn.LayerNorm(node_dim * statistics),
            nn.Linear(node_dim * statistics, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, key_dim),
        )
        self.classifier = nn.Linear(key_dim, 3)

    def _encode_nodes(
        self, groups: Mapping[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        branches = []
        if self.rgb is not None:
            rgb = groups["rgb"].float()
            if self.variant != "m0_rgb_mean":
                # Remove whole-building appearance/style while retaining each
                # region's structural deviation from its own building.
                mean = _masked_mean(rgb, mask)[:, None, :]
                variance = _masked_mean((rgb - mean).square(), mask)[:, None, :]
                rgb = (rgb - mean) / variance.add(1e-5).sqrt()
            branches.append(self.rgb(rgb))
        if self.sar is not None and self.spec.sar_group is not None:
            branches.append(self.sar(groups[self.spec.sar_group]))
        if self.relation is not None and self.spec.relation is not None:
            branches.append(self.relation(groups[self.spec.relation]))
        if self.multiring is not None:
            branches.append(self.multiring(groups["multiring"]))
        return self.node_projection(torch.cat(branches, dim=-1))

    def forward(
        self,
        groups: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = self._encode_nodes(groups, mask) * mask[:, :, None]
        geometry = groups["geometry"].float()
        position = geometry[:, :, 4:6]
        if self.spec.shuffled_geometry:
            position = _roll_valid_positions(position, mask)
        if self.relation_layer is not None:
            quality = None
            if self.spec.quality_gate:
                # quality[...,0] is mean matching entropy; turn it into reliability.
                quality = (1.0 - groups["quality"][:, :, 0].float()).clamp(0.01, 1.0)
            token = self.relation_layer(token, position, mask, quality)
        mean = _masked_mean(token, mask)
        if self.spec.aggregation == "mean":
            summary = mean
        else:
            maximum = _masked_max(token, mask)
            score = self.anomaly(token)[:, :, 0].masked_fill(~mask, -1e4)
            take = min(3, token.shape[1])
            indices = score.topk(take, dim=1).indices
            top_mask = torch.zeros_like(mask).scatter(1, indices, True) & mask
            top = _masked_mean(token, top_mask)
            median = _masked_quantile(token, mask, 0.50)
            p90 = _masked_quantile(token, mask, 0.90)
            summary = torch.cat((mean, maximum, top, median, p90), dim=1)
        key = F.normalize(self.building_projection(summary).float(), dim=1)
        return key, self.classifier(key), token


def building_groups(payload: Mapping[str, object], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        raise RuntimeError("building payload has no factorized groups")
    return {name: value[indices].to(device).float() for name, value in groups.items()}


def cross_event_contrastive(
    key: torch.Tensor,
    labels: torch.Tensor,
    events: Sequence[str],
    images: Sequence[str],
    temperature: float = 0.10,
) -> torch.Tensor:
    similarity = key @ key.t() / float(temperature)
    eye = torch.eye(key.shape[0], device=key.device, dtype=torch.bool)
    same_label = labels[:, None] == labels[None, :]
    event_diff = torch.tensor(
        [[events[i] != events[j] for j in range(len(events))] for i in range(len(events))],
        device=key.device, dtype=torch.bool,
    )
    image_diff = torch.tensor(
        [[images[i] != images[j] for j in range(len(images))] for i in range(len(images))],
        device=key.device, dtype=torch.bool,
    )
    preferred = same_label & event_diff & image_diff & ~eye
    fallback = same_label & image_diff & ~eye
    positive = torch.where(preferred.any(1, keepdim=True), preferred, fallback)
    valid = positive.any(1)
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    logits = similarity.masked_fill(eye, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probability * positive.float()).sum(1) / positive.sum(1).clamp_min(1)
    return loss[valid].mean()


def audit_contract(metadata: Mapping[str, object], split: str) -> None:
    if split not in {"train", "validation"}:
        raise RuntimeError("Test records are forbidden in V1")
    if split != "train" and bool(metadata.get("memory_write_allowed", False)):
        raise RuntimeError("Validation records cannot write Memory")
    forbidden = set(metadata.get("building_key_groups", ())) & {"p3_state", "label", "event", "image"}
    if forbidden:
        raise RuntimeError(f"building retrieval Key contains forbidden shortcuts: {sorted(forbidden)}")


__all__ = [
    "PROTOCOL", "SPECS", "VARIANTS", "BuildingRelationalEncoder", "VariantSpec",
    "audit_contract", "building_groups", "cross_event_contrastive",
]
