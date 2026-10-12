from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    _p3_state,
    _pool_feature,
    _ring,
    cross_event_supervised_contrastive,
    retrieve_diverse_topk,
)
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import valid_miou

from . import PROTOCOL


VARIANTS = ("k1_local", "k2_multiscale", "k3_structural")
VARIANT_GROUPS = {
    "k1_local": ("change_local",),
    "k2_multiscale": ("change_local", "change_semantic"),
    "k3_structural": ("rgb_structure", "change_local", "change_semantic", "multiring"),
}


def resize_feature(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False).to(value.dtype)


def multiring_change_descriptor(
    local: torch.Tensor,
    semantic: torch.Tensor,
    raw_sar: torch.Tensor,
    mask: torch.Tensor,
    radii: tuple[int, int] = (2, 4),
) -> torch.Tensor:
    """Compact core/ring evidence; independent of P3 final probabilities."""
    signals = (
        local.float().norm(dim=0),
        semantic.float().norm(dim=0),
        local.float().abs().mean(0),
        raw_sar.float(),
    )
    output = []
    for radius in radii:
        ring = _ring(mask, int(radius))
        for signal in signals:
            core = signal[mask]
            core_mean = core.mean()
            ring_mean = signal[ring].mean() if bool(ring.any()) else core_mean
            difference = core_mean - ring_mean
            top_count = max(1, int(math.ceil(core.numel() * 0.10)))
            top_anomaly = core.topk(top_count).values.mean()
            output.extend((core_mean, ring_mean, difference, difference.abs(), top_anomaly))
    return torch.stack(output).to(local.dtype)


def region_geometry(mask: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
    yy, xx = torch.where(mask)
    height, width = mask.shape
    area = mask.float().mean()
    hspan = (yy.max() - yy.min() + 1).float() / max(height, 1)
    wspan = (xx.max() - xx.min() + 1).float() / max(width, 1)
    cy = yy.float().mean() / max(height - 1, 1)
    cx = xx.float().mean() / max(width - 1, 1)
    return torch.stack((area, hspan, wspan, cy, cx, building[mask].mean()))


def extract_region_groups(
    *,
    rgb_structure: torch.Tensor,
    change_local: torch.Tensor,
    change_semantic: torch.Tensor,
    raw_sar: torch.Tensor,
    ids: torch.Tensor,
    p_building: torch.Tensor,
    p3_probability: torch.Tensor,
    target: torch.Tensor,
    minimum_area: int = 4,
    minimum_building: float = 0.50,
) -> list[dict[str, Any]]:
    """Extract local records from one image at a common feature grid."""
    size = change_local.shape[-2:]
    rgb_structure = resize_feature(rgb_structure[None], size)[0]
    change_semantic = resize_feature(change_semantic[None], size)[0]
    raw_sar = resize_feature(raw_sar[None, None], size)[0, 0]
    building = resize_feature(p_building[None, None], size)[0, 0]
    probability = resize_feature(p3_probability[None], size)[0]
    target_small = F.interpolate(
        target[None, None].float(), size=size, mode="nearest"
    )[0, 0].long()
    rows = []
    for region_id in torch.unique(ids).tolist():
        mask = ids == int(region_id)
        area = int(mask.sum())
        if area < int(minimum_area) or float(building[mask].mean()) < float(minimum_building):
            continue
        valid = mask & (target_small != 255)
        foreground = valid & (target_small >= 1) & (target_small <= 3)
        if not bool(foreground.any()):
            continue
        counts = torch.stack([(target_small[foreground] == label).sum() for label in (1, 2, 3)]).float()
        soft = counts / counts.sum().clamp_min(1.0)
        label = int(soft.argmax().item()) + 1
        purity = float(soft.max())
        p3_state, p3_label, uncertainty = _p3_state(probability, mask)
        rows.append({
            "region_id": int(region_id),
            "mask": mask,
            "groups": {
                "rgb_structure": _pool_feature(rgb_structure, mask),
                "change_local": _pool_feature(change_local, mask),
                "change_semantic": _pool_feature(change_semantic, mask),
                "multiring": multiring_change_descriptor(
                    change_local, change_semantic, raw_sar, mask
                ),
                "geometry": region_geometry(mask, building),
                "p3_state": p3_state,
            },
            "soft_label": soft,
            "label": label,
            "purity": purity,
            "area": area,
            "p3_region_label": p3_label,
            "p3_uncertainty": uncertainty,
        })
    return rows


class BranchEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 48, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


class ChangeMambaTokenEncoder(nn.Module):
    """Retrieval-specific projection; P3 state is never part of the Key."""
    def __init__(
        self, group_dims: Mapping[str, int], variant: str,
        key_dim: int = 128, hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.names = VARIANT_GROUPS[variant]
        self.branches = nn.ModuleDict({
            name: BranchEncoder(int(group_dims[name])) for name in self.names
        })
        merged = 48 * len(self.names)
        self.fuse = nn.Sequential(
            nn.LayerNorm(merged), nn.Linear(merged, hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, key_dim),
        )
        self.classifier = nn.Linear(key_dim, 3)

    def forward(self, groups: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat([self.branches[name](groups[name]) for name in self.names], dim=1)
        key = F.normalize(self.fuse(value).float(), dim=1)
        return key, self.classifier(key)


class BalancedEventClassSampler:
    def __init__(self, labels: torch.Tensor, events: Sequence[str], seed: int) -> None:
        self.rng = random.Random(int(seed))
        buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, (label, event) in enumerate(zip(labels.tolist(), events)):
            buckets[(int(label), str(event))].append(index)
        self.by_class = {
            label: {event: rows for (cls, event), rows in buckets.items() if cls == label}
            for label in (1, 2, 3)
        }
        for label, rows in self.by_class.items():
            if not rows:
                raise RuntimeError(f"class {label} has no records")

    def draw(self, batch_size: int) -> torch.Tensor:
        output = []
        for offset in range(int(batch_size)):
            label = 1 + offset % 3
            event = self.rng.choice(sorted(self.by_class[label]))
            output.append(self.rng.choice(self.by_class[label][event]))
        self.rng.shuffle(output)
        return torch.tensor(output, dtype=torch.long)


def groups_at(payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value[indices].to(device).float() for name, value in payload["groups"].items()}


def retrieval_relative_gap(result: Mapping[str, torch.Tensor], memory_labels: torch.Tensor) -> torch.Tensor:
    """Predicted-class similarity minus strongest alternative-class similarity."""
    labels = memory_labels[result["indices"]].long()
    scores = result["scores"].float()
    prediction = result["prediction"].long()
    same = labels == prediction[:, None]
    positive = scores.masked_fill(~same, -torch.inf).max(1).values
    negative = scores.masked_fill(same, -torch.inf).max(1).values
    positive = torch.where(torch.isfinite(positive), positive, scores[:, 0])
    negative = torch.where(torch.isfinite(negative), negative, scores[:, -1])
    return positive - negative


def balanced_event_folds(events: Sequence[str], count: int) -> list[set[str]]:
    sizes = defaultdict(int)
    for event in events:
        sizes[str(event)] += 1
    folds: list[set[str]] = [set() for _ in range(max(2, int(count)))]
    loads = [0 for _ in folds]
    for event, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(len(folds)), key=lambda index: (loads[index], index))
        folds[target].add(event)
        loads[target] += size
    return folds


def threshold_candidates() -> list[float]:
    """Include the safe reject-all action missing from the old verifier."""
    return [integer / 100.0 for integer in range(5, 100)] + [1.01]


__all__ = [
    "PROTOCOL", "VARIANTS", "VARIANT_GROUPS", "BalancedEventClassSampler",
    "ChangeMambaTokenEncoder", "balanced_event_folds", "cross_event_supervised_contrastive",
    "extract_region_groups", "groups_at", "retrieval_diverse_topk", "retrieval_relative_gap",
    "threshold_candidates", "valid_miou",
]

# Backward-readable alias used by the experiment scripts.
retrieval_diverse_topk = retrieve_diverse_topk
