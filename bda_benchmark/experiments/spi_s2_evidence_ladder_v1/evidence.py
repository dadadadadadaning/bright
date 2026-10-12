from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (
    RGBGuidedSoftSuperpixels,
    pool_superpixel_statistics,
    unpool_superpixel_values,
)
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import hard_local_matching_maps


@dataclass(frozen=True)
class EvidenceTensor:
    name: str
    values: torch.Tensor
    channel_names: tuple[str, ...]

    def validate(self, batch: int, size: tuple[int, int]) -> None:
        if self.values.shape != (batch, len(self.channel_names), *size):
            raise ValueError(
                f"{self.name} evidence has shape {tuple(self.values.shape)}, "
                f"expected {(batch, len(self.channel_names), *size)}"
            )
        if not bool(torch.isfinite(self.values).all()):
            raise FloatingPointError(f"{self.name} evidence contains non-finite values")


def im4cd_reliability_evidence(
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
    *,
    window_size: int = 5,
) -> EvidenceTensor:
    maps = hard_local_matching_maps(rgb_feature, sar_feature, window_size=window_size)
    names = (
        "colocated_cosine_distance",
        "one_minus_peak",
        "peak_gain",
        "hard_residual",
        "correlation_margin",
        "correlation_entropy",
        "matching_confidence",
        "confidence_weighted_residual",
    )
    values = torch.cat([maps[name].float() for name in names], dim=1)
    values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)
    result = EvidenceTensor("im4cd_translation_residual", values, names)
    result.validate(rgb_feature.shape[0], output_size)
    return result


def _local_relations(feature: torch.Tensor) -> torch.Tensor:
    feature = F.normalize(feature.float(), dim=1, eps=1e-6)
    height, width = feature.shape[-2:]
    padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
    rows = []
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        neighbour = padded[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        rows.append((feature * neighbour).sum(1, keepdim=True))
    return torch.cat(rows, dim=1)


@torch.no_grad()
def relation_difference_evidence(
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
) -> EvidenceTensor:
    if rgb_feature.shape != sar_feature.shape:
        raise ValueError("RGB/SAR relation evidence requires aligned feature shapes")
    difference = (_local_relations(rgb_feature) - _local_relations(sar_feature)).abs()
    values = torch.cat(
        (
            difference.mean(1, keepdim=True),
            difference.std(1, keepdim=True, unbiased=False),
            difference.amax(1, keepdim=True),
            torch.quantile(difference, 0.75, dim=1, keepdim=True),
        ),
        dim=1,
    )
    values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)
    names = ("relation_mean", "relation_std", "relation_max", "relation_q75")
    result = EvidenceTensor("within_modality_relation_difference", values, names)
    result.validate(rgb_feature.shape[0], output_size)
    return result


@torch.no_grad()
def cross_event_memory_evidence(
    query_keys: torch.Tensor,
    region_masks: torch.Tensor,
    bank_keys: torch.Tensor,
    bank_labels: torch.Tensor,
    bank_events: Sequence[str],
    query_event: str,
    *,
    topk: int = 5,
) -> EvidenceTensor:
    """Broadcast class-balanced, leave-one-event-out retrieval scores to regions."""

    if query_keys.ndim != 2 or bank_keys.ndim != 2 or query_keys.shape[1] != bank_keys.shape[1]:
        raise ValueError("query/bank keys must be compatible [R,D] and [N,D]")
    if region_masks.ndim != 3 or region_masks.shape[0] != query_keys.shape[0]:
        raise ValueError("region_masks must be [R,H,W]")
    if len(bank_events) != bank_keys.shape[0] or bank_labels.numel() != bank_keys.shape[0]:
        raise ValueError("memory metadata does not align")
    query = F.normalize(query_keys.float(), dim=1)
    bank = F.normalize(bank_keys.float().to(query.device), dim=1)
    event_ok = torch.tensor([str(event) != str(query_event) for event in bank_events], device=bank.device)
    per_class = []
    availability = []
    for label in (1, 2, 3):
        selected = event_ok & (bank_labels.to(bank.device).long() == label)
        if int(selected.sum()) == 0:
            per_class.append(query.new_full((query.shape[0], 2), -1.0))
            availability.append(query.new_zeros((query.shape[0], 1)))
            continue
        similarity = query @ bank[selected].T
        count = min(int(topk), similarity.shape[1])
        peaks = similarity.topk(count, dim=1).values
        per_class.append(torch.stack((peaks[:, 0], peaks.mean(1)), dim=1))
        availability.append(query.new_ones((query.shape[0], 1)))
    scores = torch.cat(per_class, dim=1)
    available = torch.cat(availability, dim=1)
    class_best = scores[:, (0, 2, 4)]
    top2 = class_best.topk(2, dim=1).values
    region_values = torch.cat((scores, available, top2[:, :1] - top2[:, 1:2]), dim=1)
    height, width = region_masks.shape[-2:]
    output = query.new_zeros((1, region_values.shape[1], height, width))
    for index, mask in enumerate(region_masks.to(output.device).bool()):
        output[0, :, mask] = region_values[index, :, None]
    names = (
        "intact_top1", "intact_topk_mean", "damaged_top1", "damaged_topk_mean",
        "destroyed_top1", "destroyed_topk_mean", "intact_available", "damaged_available",
        "destroyed_available", "retrieval_margin",
    )
    result = EvidenceTensor("cross_event_memory", output, names)
    result.validate(1, (height, width))
    return result


@torch.no_grad()
def superpixel_partition_evidence(
    pre_rgb: torch.Tensor,
    feature: torch.Tensor,
    output_size: tuple[int, int],
    *,
    mode: str,
    cell_size: int = 2,
) -> EvidenceTensor:
    """Matched RGB-superpixel/grid diagnostic; both branches have equal node budget."""

    partition = RGBGuidedSoftSuperpixels(cell_size=cell_size).to(pre_rgb.device)
    rgb_low = F.interpolate(pre_rgb.float(), size=feature.shape[-2:], mode="bilinear", align_corners=False)
    assignment = partition(rgb_low, mode)
    mean, std, _maximum, mass = pool_superpixel_statistics(feature.float(), assignment)
    reconstruction = unpool_superpixel_values(mean, assignment, feature.shape[-2:])
    spread = unpool_superpixel_values(std.mean(-1, keepdim=True), assignment, feature.shape[-2:])
    mass_map = unpool_superpixel_values(mass, assignment, feature.shape[-2:])
    reconstruction_error = (feature.float() - reconstruction).square().mean(1, keepdim=True).sqrt()
    mass_map = mass_map / mass_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    values = torch.cat((assignment.entropy, reconstruction_error, spread, mass_map), dim=1)
    values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)
    names = ("assignment_entropy", "reconstruction_error", "within_region_spread", "normalized_mass")
    result = EvidenceTensor(f"soft_superpixel_{mode}", values, names)
    result.validate(pre_rgb.shape[0], output_size)
    return result


def concatenate_evidence(
    groups: Mapping[str, EvidenceTensor],
    ordered_names: Sequence[str],
    *,
    batch: int,
    size: tuple[int, int],
) -> torch.Tensor:
    values = []
    for name in ordered_names:
        if name not in groups:
            raise KeyError(f"missing evidence group: {name}")
        groups[name].validate(batch, size)
        values.append(groups[name].values)
    return torch.cat(values, dim=1) if values else torch.empty(batch, 0, *size)


@torch.no_grad()
def mean_region_keys(feature: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalized mean keys without labels or trainable projections."""

    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("mean_region_keys currently expects one [1,C,h,w] feature map")
    if region_masks.ndim != 3:
        raise ValueError("region_masks must be [R,H,W]")
    masks = F.interpolate(
        region_masks[:, None].float().to(feature.device),
        size=feature.shape[-2:],
        mode="area",
    )[:, 0]
    flat_feature = feature[0].float().reshape(feature.shape[1], -1)
    rows = []
    for mask in masks:
        weight = mask.reshape(-1).clamp_min(0.0)
        if float(weight.sum()) <= 1e-6:
            raise RuntimeError("region vanished at feature resolution")
        rows.append((flat_feature * weight[None]).sum(1) / weight.sum())
    return F.normalize(torch.stack(rows), dim=1, eps=1e-8)

