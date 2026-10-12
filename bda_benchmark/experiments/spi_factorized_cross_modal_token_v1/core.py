from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1"
CLASSES = (1, 2, 3)
VARIANTS = ("t0_concat", "t1_factorized", "t2_aligned", "t3_multiring")
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
REFERENCE_FEATURE_AREA = 160 * 160


@dataclass(frozen=True)
class FactorizedRegionRecord:
    rgb: torch.Tensor
    sar: torch.Tensor
    relation_unaligned: torch.Tensor
    relation_aligned: torch.Tensor
    multiring: torch.Tensor
    geometry: torch.Tensor
    quality: torch.Tensor
    p3_state: torch.Tensor
    soft_label: torch.Tensor
    hard_label: int
    purity: float
    building_overlap: float
    area: int
    event_id: str
    image_id: str
    region_id: int
    component_id: int
    p3_region_label: int


def _grid_ids(height: int, width: int, n_segments: int, device: torch.device) -> torch.Tensor:
    side = max(2, int(math.sqrt(height * width / max(int(n_segments), 1))))
    columns = (width + side - 1) // side
    yy = torch.arange(height, device=device)[:, None] // side
    xx = torch.arange(width, device=device)[None] // side
    return (yy * columns + xx).long()


def make_region_ids(
    pre_rgb: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
    n_segments: int,
    compactness: float,
) -> torch.Tensor:
    """Generate grid/SLIC/SLICO controls from pre-disaster RGB only."""
    if mode not in {"grid", "slic", "slico", "shuffled_slico"}:
        raise ValueError(f"unsupported region mode: {mode}")
    batch, _, _, _ = pre_rgb.shape
    height, width = size
    scaled = max(2, int(round(float(n_segments) * height * width / REFERENCE_FEATURE_AREA)))
    if mode == "grid":
        return _grid_ids(height, width, scaled, pre_rgb.device)[None].expand(batch, -1, -1).contiguous()
    try:
        from skimage.segmentation import slic
    except ImportError as error:
        raise RuntimeError("scikit-image is required for SLIC/SLICO") from error
    source = F.interpolate(pre_rgb.float(), size=size, mode="bilinear", align_corners=False)
    mean = source.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = source.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    source = ((source * std + mean) / 255.0).clamp(0.0, 1.0)
    labels = []
    for image in source:
        if mode == "shuffled_slico":
            image = torch.flip(image, dims=(-1,))
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        value = slic(
            array,
            n_segments=scaled,
            compactness=float(compactness),
            start_label=0,
            channel_axis=-1,
            convert2lab=True,
            slic_zero=mode in {"slico", "shuffled_slico"},
        ).astype(np.int64, copy=False)
        labels.append(torch.from_numpy(value).to(pre_rgb.device).long())
    return torch.stack(labels)


def split_regions_by_building(ids: torch.Tensor, p_building: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if p_building.ndim == 4:
        p_building = p_building[:, 0]
    if ids.shape != p_building.shape:
        p_building = F.interpolate(
            p_building[:, None].float(), size=ids.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    # The odd IDs are predicted-building parts; even IDs are background parts.
    return ids * 2 + (p_building >= float(threshold)).long()


def predicted_component_ids(p_building: torch.Tensor, size: tuple[int, int], threshold: float = 0.5) -> torch.Tensor:
    """Connected-component provenance for later building-bag experiments."""
    try:
        from skimage.measure import label
    except ImportError as error:
        raise RuntimeError("scikit-image is required for component provenance") from error
    value = F.interpolate(p_building.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    output = []
    for row in value:
        component = label((row >= float(threshold)).cpu().numpy(), connectivity=1)
        output.append(torch.from_numpy(component.astype(np.int64, copy=False)).to(row.device))
    return torch.stack(output).long()


def local_alignment_diagnostics(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    radius: int = 1,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Locally align SAR and return confidence maps separate from semantic content."""
    if rgb.shape != sar.shape or rgb.ndim != 4:
        raise ValueError("rgb and sar must share [B,C,H,W]")
    if radius < 0 or temperature <= 0:
        raise ValueError("radius must be non-negative and temperature positive")
    batch, channels, height, width = rgb.shape
    if radius == 0:
        one = rgb.new_ones((batch, 1, height, width))
        zero = rgb.new_zeros((batch, 1, height, width))
        return sar, {
            "entropy": zero,
            "expected_offset": zero,
            "expected_dx": zero,
            "expected_dy": zero,
            "max_probability": one,
            "match_margin": one,
        }
    kernel = 2 * radius + 1
    count = kernel * kernel
    patches = F.unfold(sar, kernel_size=kernel, padding=radius).view(
        batch, channels, count, height, width
    )
    query = F.normalize(rgb.float(), dim=1)
    candidates = F.normalize(patches.float(), dim=1)
    scores = (query.unsqueeze(2) * candidates).sum(1) / float(temperature)
    attention = scores.softmax(1)
    aligned = (attention.unsqueeze(1) * patches.float()).sum(2).to(rgb.dtype)
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(1, keepdim=True)
    entropy = entropy / math.log(float(count))
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    dx = attention.new_tensor([item[0] for item in offsets]).view(1, count, 1, 1)
    dy = attention.new_tensor([item[1] for item in offsets]).view(1, count, 1, 1)
    distance = (dx.square() + dy.square()).sqrt()
    top2 = attention.topk(min(2, count), dim=1).values
    margin = top2[:, :1] if count == 1 else top2[:, :1] - top2[:, 1:2]
    diagnostics = {
        "entropy": entropy.to(rgb.dtype),
        "expected_offset": (attention * distance).sum(1, keepdim=True).to(rgb.dtype),
        "expected_dx": (attention * dx).sum(1, keepdim=True).to(rgb.dtype),
        "expected_dy": (attention * dy).sum(1, keepdim=True).to(rgb.dtype),
        "max_probability": top2[:, :1].to(rgb.dtype),
        "match_margin": margin.to(rgb.dtype),
    }
    return aligned, diagnostics


def recover_sar_display(post: torch.Tensor, stats: Mapping[str, object] | None = None) -> torch.Tensor:
    """Recover the replicated display SAR channel and apply frozen train-only stats."""
    display = post[:, :1].float() * float(IMAGENET_STD[0]) + float(IMAGENET_MEAN[0])
    if stats is None:
        return display / 255.0
    normal = stats.get("normalization", stats)
    low = float(normal.get("p01", 0.0))
    high = float(normal.get("p99", 255.0))
    mean = float(normal.get("mean", 0.0))
    std = max(float(normal.get("std", 1.0)), 1e-6)
    return (display.clamp(low, high) - mean) / std


def rgb_orientation_maps(pre_rgb: torch.Tensor, size: tuple[int, int], bins: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    gray = pre_rgb.float().mean(1, keepdim=True)
    kernel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    gx = F.interpolate(gx, size=size, mode="bilinear", align_corners=False)
    gy = F.interpolate(gy, size=size, mode="bilinear", align_corners=False)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
    scale = magnitude.flatten(2).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    magnitude = (magnitude / scale).clamp(0.0, 5.0)
    angle = torch.remainder(torch.atan2(gy, gx), math.pi)
    index = torch.floor(angle / math.pi * bins).long().clamp(0, bins - 1)
    histogram = magnitude.new_zeros((pre_rgb.shape[0], bins, *size))
    histogram.scatter_(1, index, magnitude)
    return magnitude, histogram


def _erode(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    value = mask.float()[None, None]
    kernel = 2 * radius + 1
    return (-F.max_pool2d(-value, kernel, stride=1, padding=radius)[0, 0]) > 0.5


def _ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
    value = mask.float()[None, None]
    dilated = F.max_pool2d(value, 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0.5
    return dilated & ~mask


def _perimeter(mask: torch.Tensor) -> torch.Tensor:
    return (mask & ~_erode(mask, 1)).float().sum()


def _attention_pool(values: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(score.float(), dim=0).to(values.dtype)
    return (weights[:, None] * values).sum(0)


def _pool_feature(feature: torch.Tensor, mask: torch.Tensor, score: torch.Tensor | None = None) -> torch.Tensor:
    values = feature.permute(1, 2, 0)[mask]
    if values.numel() == 0:
        return feature.new_zeros(feature.shape[0] * 4)
    if score is None:
        score = values.float().norm(dim=1)
    return torch.cat(
        (
            values.mean(0),
            values.float().std(0, unbiased=False).to(values.dtype),
            values.max(0).values,
            _attention_pool(values, score),
        )
    )


def _scalar_stats(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(6)
    flat = values.float().flatten()
    quantile = torch.quantile(flat, flat.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    top_count = max(1, int(math.ceil(flat.numel() * 0.10)))
    return torch.cat((quantile, flat.topk(top_count).values.mean()[None])).to(values.dtype)


def _three_stats(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(3)
    return torch.stack((values.mean(), values.float().std(unbiased=False).to(values.dtype), values.max()))


def _relation_descriptor(rgb: torch.Tensor, sar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    difference = rgb - sar
    absolute = difference.abs()
    product = rgb * sar
    cosine = F.cosine_similarity(rgb.float(), sar.float(), dim=0).to(rgb.dtype)
    ratio = rgb.float().norm(dim=0) / sar.float().norm(dim=0).clamp_min(1e-6)
    values = absolute.permute(1, 2, 0)[mask]
    signed = difference.permute(1, 2, 0)[mask]
    multiplied = product.permute(1, 2, 0)[mask]
    score = values.float().mean(1)
    return torch.cat(
        (
            values.mean(0),
            values.float().std(0, unbiased=False).to(values.dtype),
            values.max(0).values,
            _attention_pool(values, score),
            signed.mean(0),
            multiplied.mean(0),
            _three_stats(cosine[mask]),
            _three_stats(ratio[mask].to(rgb.dtype)),
        )
    )


def _fragmentation(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:, :] & mask[:-1, :]
    transitions = (prediction[:, 1:] != prediction[:, :-1])[horizontal].float().sum()
    transitions += (prediction[1:, :] != prediction[:-1, :])[vertical].float().sum()
    adjacency = horizontal.float().sum() + vertical.float().sum()
    fractions = torch.stack([(prediction[mask] == value).float().mean() for value in range(4)])
    class_entropy = -(fractions * fractions.clamp_min(1e-8).log()).sum() / math.log(4.0)
    return torch.stack((transitions / adjacency.clamp_min(1.0), class_entropy))


def _p3_state(probability: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    values = probability.permute(1, 2, 0)[mask]
    prediction = probability.argmax(0)
    entropy_map = -(probability * probability.clamp_min(1e-8).log()).sum(0) / math.log(4.0)
    occurrence_margin = probability[2:4].sum(0) - probability[1]
    fractions = torch.stack([(prediction[mask] == value).float().mean() for value in range(4)])
    entropy_values = entropy_map[mask]
    margin_values = occurrence_margin[mask]
    state = torch.cat(
        (
            values.mean(0),
            values.float().std(0, unbiased=False).to(values.dtype),
            torch.stack((entropy_values.mean(), entropy_values.std(unbiased=False), torch.quantile(entropy_values, 0.90))),
            torch.stack((margin_values.mean(), margin_values.std(unbiased=False), margin_values.min())),
            fractions,
            _fragmentation(prediction, mask),
        )
    )
    majority = int(fractions.argmax().item())
    uncertainty = float(1.0 - margin_values.abs().mean().clamp(0.0, 1.0).item())
    return state, majority, uncertainty


def _multiring_descriptor(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    aligned_sar: torch.Tensor,
    raw_sar: torch.Tensor,
    mask: torch.Tensor,
    radii: tuple[int, int],
) -> torch.Tensor:
    absolute = (rgb - aligned_sar).abs().mean(0)
    signals = (
        rgb.float().norm(dim=0).to(rgb.dtype),
        sar.float().norm(dim=0).to(rgb.dtype),
        absolute,
        F.cosine_similarity(rgb.float(), aligned_sar.float(), dim=0).to(rgb.dtype),
        raw_sar,
    )
    output = []
    for radius in radii:
        ring = _ring(mask, radius)
        for signal in signals:
            core_mean = signal[mask].mean()
            ring_mean = signal[ring].mean() if bool(ring.any()) else core_mean
            difference = core_mean - ring_mean
            output.extend((core_mean, ring_mean, difference, difference.abs()))
    return torch.stack(output)


def extract_factorized_records(
    *,
    rgb: torch.Tensor,
    sar: torch.Tensor,
    aligned_sar: torch.Tensor,
    raw_sar: torch.Tensor,
    ids: torch.Tensor,
    components: torch.Tensor,
    p_building: torch.Tensor,
    damage_probability: torch.Tensor,
    diagnostics: Mapping[str, torch.Tensor],
    edge: torch.Tensor,
    orientation: torch.Tensor,
    damage: torch.Tensor | None,
    location: torch.Tensor | None,
    names: Sequence[str],
    events: Sequence[str],
    min_query_building: float = 0.60,
    min_write_building: float = 0.70,
    min_purity: float = 0.80,
    min_tokens: int = 4,
    ring_radii: tuple[int, int] = (2, 4),
    require_reliable_label: bool = True,
) -> list[FactorizedRegionRecord]:
    size = rgb.shape[-2:]
    if not (rgb.shape == sar.shape == aligned_sar.shape):
        raise ValueError("RGB, SAR and aligned SAR maps must match")
    if ids.shape != rgb.shape[:1] + size:
        raise ValueError("region IDs must match feature grid")
    raw_sar = F.interpolate(raw_sar.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    raw_sar_p90 = torch.quantile(raw_sar.flatten(1), 0.90, dim=1)
    p_building = F.interpolate(p_building.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    probability = F.interpolate(damage_probability.float(), size=size, mode="bilinear", align_corners=False)
    edge = F.interpolate(edge.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    orientation = F.interpolate(orientation.float(), size=size, mode="bilinear", align_corners=False)
    diagnostic_maps = {
        key: F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
        for key, value in diagnostics.items()
    }
    if damage is not None and location is not None:
        damage_low = F.interpolate(damage[:, None].float(), size=size, mode="nearest")[:, 0].long()
        location_low = F.interpolate(location[:, None].float(), size=size, mode="nearest")[:, 0].long()
    else:
        damage_low = location_low = None
    height, width = size
    yy, xx = torch.meshgrid(
        torch.arange(height, device=rgb.device),
        torch.arange(width, device=rgb.device),
        indexing="ij",
    )
    records: list[FactorizedRegionRecord] = []
    for batch_index in range(rgb.shape[0]):
        for region_id in torch.unique(ids[batch_index]).tolist():
            mask = ids[batch_index] == int(region_id)
            area = int(mask.sum().item())
            if area < int(min_tokens):
                continue
            predicted_overlap = float(p_building[batch_index][mask].mean().item())
            if predicted_overlap < float(min_query_building):
                continue
            core = _erode(mask, 1)
            if not bool(core.any()):
                core = mask
            boundary = mask & ~core
            rgb_descriptor = torch.cat(
                (
                    _pool_feature(rgb[batch_index], mask),
                    orientation[batch_index, :, mask].sum(1)
                    / orientation[batch_index, :, mask].sum().clamp_min(1e-6),
                    torch.stack((edge[batch_index][mask].mean(), edge[batch_index][boundary].mean() if bool(boundary.any()) else edge[batch_index][mask].mean())),
                )
            )
            sar_descriptor = torch.cat(
                (
                    _pool_feature(sar[batch_index], mask),
                    _scalar_stats(raw_sar[batch_index][mask]),
                    torch.stack((
                        raw_sar[batch_index][mask].topk(max(1, int(math.ceil(area * 0.05)))).values.mean(),
                        raw_sar[batch_index][mask].topk(max(1, int(math.ceil(area * 0.10)))).values.mean(),
                        (raw_sar[batch_index][mask] > raw_sar_p90[batch_index]).float().mean(),
                    )),
                )
            )
            relation_unaligned = _relation_descriptor(rgb[batch_index], sar[batch_index], mask)
            relation_aligned = _relation_descriptor(rgb[batch_index], aligned_sar[batch_index], mask)
            multiring = _multiring_descriptor(
                rgb[batch_index], sar[batch_index], aligned_sar[batch_index], raw_sar[batch_index], mask, ring_radii
            )
            perimeter = _perimeter(mask)
            y_values, x_values = yy[mask].float(), xx[mask].float()
            bbox_height = y_values.max() - y_values.min() + 1.0
            bbox_width = x_values.max() - x_values.min() + 1.0
            compactness = (perimeter.square() / (4.0 * math.pi * max(area, 1))).clamp_max(100.0)
            geometry = rgb.new_tensor(
                [
                    area / float(height * width),
                    float(perimeter.item()) / max(float(area), 1.0),
                    float(compactness.item()),
                    float((bbox_width / bbox_height.clamp_min(1.0)).item()),
                    float(x_values.mean().item()) / max(width - 1, 1),
                    float(y_values.mean().item()) / max(height - 1, 1),
                    predicted_overlap,
                    float(_ring(mask, ring_radii[0]).sum().item()) / max(area, 1),
                    float(_ring(mask, ring_radii[1]).sum().item()) / max(area, 1),
                ]
            )
            quality = torch.stack(
                (
                    diagnostic_maps["entropy"][batch_index][mask].mean(),
                    diagnostic_maps["entropy"][batch_index][mask].std(unbiased=False),
                    diagnostic_maps["expected_offset"][batch_index][mask].mean(),
                    diagnostic_maps["expected_offset"][batch_index][mask].max(),
                    diagnostic_maps["expected_dx"][batch_index][mask].mean(),
                    diagnostic_maps["expected_dy"][batch_index][mask].mean(),
                    diagnostic_maps["max_probability"][batch_index][mask].mean(),
                    diagnostic_maps["match_margin"][batch_index][mask].mean(),
                    p_building[batch_index][mask].mean(),
                    p_building[batch_index][mask].std(unbiased=False),
                )
            ).to(rgb.dtype)
            p3_state, p3_label, _ = _p3_state(probability[batch_index], mask)
            component_values = components[batch_index][mask]
            component_values = component_values[component_values > 0]
            component_id = int(torch.mode(component_values).values.item()) if component_values.numel() else 0
            if damage_low is None or location_low is None:
                soft = rgb.new_zeros(3)
                label, purity, gt_overlap = 0, 0.0, 0.0
            else:
                valid = mask & (location_low[batch_index] > 0) & (damage_low[batch_index] != 255)
                gt_overlap = float(valid.sum().item()) / float(area)
                counts = torch.stack([(damage_low[batch_index][valid] == value).float().sum() for value in CLASSES])
                soft = counts / counts.sum().clamp_min(1.0)
                purity_value, label_index = soft.max(0)
                purity = float(purity_value.item())
                label = int(label_index.item()) + 1
            if require_reliable_label and (
                label == 0 or gt_overlap < float(min_write_building) or purity < float(min_purity)
            ):
                continue
            records.append(
                FactorizedRegionRecord(
                    rgb=rgb_descriptor,
                    sar=sar_descriptor,
                    relation_unaligned=relation_unaligned,
                    relation_aligned=relation_aligned,
                    multiring=multiring,
                    geometry=geometry,
                    quality=quality,
                    p3_state=p3_state,
                    soft_label=soft,
                    hard_label=label,
                    purity=purity,
                    building_overlap=gt_overlap,
                    area=area,
                    event_id=str(events[batch_index]),
                    image_id=str(names[batch_index]),
                    region_id=int(region_id),
                    component_id=component_id,
                    p3_region_label=p3_label,
                )
            )
    return records


class BranchEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.float())


class FactorizedKeyEncoder(nn.Module):
    """T0 control or K2 factorized content Key. P3/quality never enter T1-T3."""

    def __init__(
        self,
        group_dims: Mapping[str, int],
        variant: str,
        key_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.group_dims = {str(key): int(value) for key, value in group_dims.items()}
        if variant == "t0_concat":
            names = ("rgb", "sar", "relation_aligned", "multiring", "geometry", "quality", "p3_state")
            total = sum(self.group_dims[name] for name in names)
            self.concat_names = names
            self.concat_encoder = BranchEncoder(total, key_dim, hidden_dim)
        else:
            relation_name = "relation_unaligned" if variant == "t1_factorized" else "relation_aligned"
            relation_dim = self.group_dims[relation_name]
            if variant == "t3_multiring":
                relation_dim += self.group_dims["multiring"]
            self.relation_name = relation_name
            self.rgb_encoder = BranchEncoder(self.group_dims["rgb"], 32, 128)
            self.sar_encoder = BranchEncoder(self.group_dims["sar"], 32, 128)
            self.relation_encoder = BranchEncoder(relation_dim, 48, 192)
            self.key_encoder = nn.Sequential(
                nn.LayerNorm(112),
                nn.Linear(112, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, key_dim),
            )
        self.classifier = nn.Linear(key_dim, 3)

    def forward(self, groups: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "t0_concat":
            embedding = self.concat_encoder(torch.cat([groups[name] for name in self.concat_names], dim=1))
        else:
            relation = groups[self.relation_name]
            if self.variant == "t3_multiring":
                relation = torch.cat((relation, groups["multiring"]), dim=1)
            embedding = self.key_encoder(
                torch.cat(
                    (
                        self.rgb_encoder(groups["rgb"]),
                        self.sar_encoder(groups["sar"]),
                        self.relation_encoder(relation),
                    ),
                    dim=1,
                )
            )
        key = F.normalize(embedding.float(), dim=1)
        return key, self.classifier(key)


class BuildingDeepSetEncoder(nn.Module):
    """Isolated T4 building-bag block; enabled only after T3 passes."""

    def __init__(self, token_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
        self.attention = nn.Linear(token_dim, 1)
        self.rho = nn.Sequential(nn.Linear(token_dim * 4, 256), nn.GELU(), nn.Linear(256, output_dim))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, topk: int = 3) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,N,D]")
        hidden = self.phi(tokens.float())
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError("mask must be [B,N]")
        negative = torch.finfo(hidden.dtype).min
        masked = hidden.masked_fill(~mask[:, :, None], negative)
        maximum = masked.max(1).values
        count = mask.sum(1, keepdim=True).clamp_min(1).to(hidden.dtype)
        mean = (hidden * mask[:, :, None]).sum(1) / count
        scores = self.attention(hidden)[:, :, 0].masked_fill(~mask, negative)
        weights = scores.softmax(1)
        attended = (weights[:, :, None] * hidden).sum(1)
        take = min(max(int(topk), 1), tokens.shape[1])
        top_indices = scores.topk(take, dim=1).indices
        top_mask = torch.zeros_like(mask).scatter(1, top_indices, True) & mask
        top = (hidden * top_mask[:, :, None]).sum(1) / top_mask.sum(1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return F.normalize(self.rho(torch.cat((mean, maximum, attended, top), dim=1)), dim=1)


def cross_event_supervised_contrastive(
    key: torch.Tensor,
    label: torch.Tensor,
    event_ids: Sequence[str],
    image_ids: Sequence[str],
    temperature: float = 0.10,
) -> torch.Tensor:
    similarity = key @ key.t() / float(temperature)
    eye = torch.eye(key.shape[0], device=key.device, dtype=torch.bool)
    event_diff = torch.tensor(
        [[event_ids[i] != event_ids[j] for j in range(len(event_ids))] for i in range(len(event_ids))],
        device=key.device,
        dtype=torch.bool,
    )
    image_diff = torch.tensor(
        [[image_ids[i] != image_ids[j] for j in range(len(image_ids))] for i in range(len(image_ids))],
        device=key.device,
        dtype=torch.bool,
    )
    same = label[:, None] == label[None, :]
    preferred = same & event_diff & image_diff & ~eye
    fallback = same & image_diff & ~eye
    positive = torch.where(preferred.any(1, keepdim=True), preferred, fallback)
    valid = positive.any(1)
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    logits = similarity.masked_fill(eye, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probability * positive.float()).sum(1) / positive.sum(1).clamp_min(1)
    return loss[valid].mean()


def standardize_geometry(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return F.normalize((value.float() - mean.float()) / std.float().clamp_min(1e-6), dim=1)


def retrieve_diverse_topk(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    query_geometry: torch.Tensor | None = None,
    memory_geometry: torch.Tensor | None = None,
    geometry_weight: float = 0.0,
    query_images: Sequence[str] | None = None,
    memory_images: Sequence[str] | None = None,
    query_events: Sequence[str] | None = None,
    memory_events: Sequence[str] | None = None,
    exclude_same_image: bool = True,
    exclude_same_event: bool = False,
    topk: int = 8,
    max_per_event: int = 2,
    temperature: float = 0.10,
) -> dict[str, torch.Tensor]:
    query = F.normalize(query.float(), dim=1)
    bank = F.normalize(memory_keys.float(), dim=1)
    similarity = query @ bank.t()
    if float(geometry_weight) != 0.0:
        if query_geometry is None or memory_geometry is None:
            raise ValueError("geometry tensors are required when geometry_weight is non-zero")
        similarity = similarity + float(geometry_weight) * (query_geometry @ memory_geometry.t())
    # Apply provenance exclusions before top-k.  Filtering only a short top-k
    # list afterwards can incorrectly report "no neighbour" when one event
    # dominates the closest part of the bank.
    memory_image_codes = query_image_codes = None
    if exclude_same_image and query_images is not None and memory_images is not None:
        vocabulary = {value: index for index, value in enumerate(sorted(set(map(str, memory_images)) | set(map(str, query_images))))}
        memory_image_codes = torch.tensor([vocabulary[str(value)] for value in memory_images], device=query.device)
        query_image_codes = torch.tensor([vocabulary[str(value)] for value in query_images], device=query.device)
    memory_event_codes = query_event_codes = None
    if exclude_same_event and query_events is not None and memory_events is not None:
        vocabulary = {value: index for index, value in enumerate(sorted(set(map(str, memory_events)) | set(map(str, query_events))))}
        memory_event_codes = torch.tensor([vocabulary[str(value)] for value in memory_events], device=query.device)
        query_event_codes = torch.tensor([vocabulary[str(value)] for value in query_events], device=query.device)
    for row in range(query.shape[0]):
        invalid = torch.zeros(memory_keys.shape[0], device=query.device, dtype=torch.bool)
        if memory_image_codes is not None and query_image_codes is not None:
            invalid |= memory_image_codes == query_image_codes[row]
        if memory_event_codes is not None and query_event_codes is not None:
            invalid |= memory_event_codes == query_event_codes[row]
        similarity[row].masked_fill_(invalid, -torch.inf)
    take_candidates = min(memory_keys.shape[0], max(int(topk) * 50, int(topk)))
    all_scores, all_indices = similarity.topk(take_candidates, dim=1)
    selected_indices, selected_scores, source_counts, source_entropies = [], [], [], []
    for row in range(query.shape[0]):
        chosen, scores, per_event = [], [], {}
        for score, index in zip(all_scores[row].tolist(), all_indices[row].tolist()):
            if not math.isfinite(score):
                continue
            event = str(memory_events[index]) if memory_events is not None else "unknown"
            if per_event.get(event, 0) >= int(max_per_event):
                continue
            chosen.append(index)
            scores.append(score)
            per_event[event] = per_event.get(event, 0) + 1
            if len(chosen) >= int(topk):
                break
        if not chosen:
            raise RuntimeError("retrieval has no eligible neighbour after provenance exclusions")
        while len(chosen) < int(topk):
            chosen.append(chosen[-1])
            scores.append(scores[-1])
        counts = torch.tensor(list(per_event.values()), dtype=torch.float32)
        probability = counts / counts.sum().clamp_min(1.0)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
        if counts.numel() > 1:
            entropy = entropy / math.log(float(counts.numel()))
        selected_indices.append(chosen)
        selected_scores.append(scores)
        source_counts.append(len(per_event))
        source_entropies.append(float(entropy.item()))
    indices = torch.tensor(selected_indices, device=query.device, dtype=torch.long)
    scores = torch.tensor(selected_scores, device=query.device, dtype=query.dtype)
    neighbour_labels = memory_labels[indices].long() - 1
    weight = torch.softmax(scores / float(temperature), dim=1)
    posterior = query.new_zeros((query.shape[0], 3))
    posterior.scatter_add_(1, neighbour_labels, weight)
    top2 = posterior.topk(2, dim=1).values
    prediction = posterior.argmax(1) + 1
    agreement = (neighbour_labels == (prediction - 1)[:, None]).float().mean(1)
    return {
        "posterior": posterior,
        "prediction": prediction,
        "margin": top2[:, 0] - top2[:, 1],
        "agreement": agreement,
        "max_similarity": scores[:, 0],
        "indices": indices,
        "scores": scores,
        "source_event_count": torch.tensor(source_counts, device=query.device),
        "source_event_entropy": torch.tensor(source_entropies, device=query.device),
    }


def groups_from_payload(payload: Mapping[str, object], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    groups = payload["groups"]
    if not isinstance(groups, Mapping):
        raise TypeError("record payload groups must be a mapping")
    return {name: value[indices].to(device).float() for name, value in groups.items()}


__all__ = [
    "PROTOCOL",
    "VARIANTS",
    "BuildingDeepSetEncoder",
    "FactorizedKeyEncoder",
    "FactorizedRegionRecord",
    "cross_event_supervised_contrastive",
    "extract_factorized_records",
    "groups_from_payload",
    "local_alignment_diagnostics",
    "make_region_ids",
    "predicted_component_ids",
    "recover_sar_display",
    "retrieve_diverse_topk",
    "rgb_orientation_maps",
    "split_regions_by_building",
    "standardize_geometry",
]
