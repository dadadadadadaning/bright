from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2"
CLASSES = (1, 2, 3)  # Intact, Damaged, Destroyed. Background is never stored.


@dataclass(frozen=True)
class RegionRecord:
    descriptor: torch.Tensor
    soft_label: torch.Tensor
    hard_label: int
    purity: float
    building_overlap: float
    area: int
    event_id: str
    image_id: str
    # Spatial provenance is required by downstream read-only audits that map a
    # retrieved region proposal back to the P3 pixel grid.  Defaults preserve
    # compatibility with the already-saved V2 record caches.
    region_id: int = -1
    batch_index: int = -1


def local_tolerant_alignment(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    radius: int = 1,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align SAR locally to RGB without assuming exact pixel correspondence.

    Returns aligned SAR, normalized attention entropy and expected offset
    magnitude.  Both inputs must already be modality-specific projected maps.
    """
    if rgb.shape != sar.shape or rgb.ndim != 4:
        raise ValueError("rgb and sar must share [B,C,H,W]")
    if radius < 0 or temperature <= 0:
        raise ValueError("radius must be non-negative and temperature positive")
    if radius == 0:
        zero = rgb.new_zeros((rgb.shape[0], 1, *rgb.shape[-2:]))
        return sar, zero, zero
    b, c, h, w = rgb.shape
    kernel = 2 * radius + 1
    count = kernel * kernel
    patches = F.unfold(sar, kernel_size=kernel, padding=radius)
    patches = patches.view(b, c, count, h, w)
    query = F.normalize(rgb.float(), dim=1)
    keys = F.normalize(patches.float(), dim=1)
    score = (query.unsqueeze(2) * keys).sum(1) / float(temperature)
    attention = score.softmax(1)
    aligned = (attention.unsqueeze(1) * patches.float()).sum(2).to(rgb.dtype)
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(1, keepdim=True)
    entropy = entropy / math.log(float(count))
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append(math.sqrt(float(dx * dx + dy * dy)))
    distance = attention.new_tensor(offsets).view(1, count, 1, 1)
    expected_offset = (attention * distance).sum(1, keepdim=True)
    return aligned, entropy.to(rgb.dtype), expected_offset.to(rgb.dtype)


def relation_map(
    rgb: torch.Tensor,
    aligned_sar: torch.Tensor,
    fused: torch.Tensor,
    cross_token: nn.Module,
) -> torch.Tensor:
    """Use the trained MICB relation construction, retaining signed evidence."""
    if rgb.shape != aligned_sar.shape or rgb.shape != fused.shape:
        raise ValueError("projected rgb, aligned sar and fused maps must match")
    base = torch.cat(
        (rgb, aligned_sar, (rgb - aligned_sar).abs(), rgb * aligned_sar, fused),
        dim=1,
    )
    # The signed discrepancy is supplied as a bounded residual.  This avoids
    # changing the input contract of the already-trained P3 cross_token block.
    relation = cross_token(base)
    return relation + 0.10 * torch.tanh(rgb - aligned_sar)


def region_boundary(ids: torch.Tensor) -> torch.Tensor:
    if ids.ndim != 3:
        raise ValueError("ids must be [B,H,W]")
    edge = torch.zeros_like(ids, dtype=torch.bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :, 1:] |= ids[:, :, 1:] != ids[:, :, :-1]
    edge[:, :, :-1] |= ids[:, :, 1:] != ids[:, :, :-1]
    return edge


def rgb_sobel_edge(pre_rgb: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Label-free pre-disaster structural boundary strength."""
    gray = pre_rgb.float().mean(1, keepdim=True)
    kernel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
    edge = F.interpolate(edge, size=size, mode="bilinear", align_corners=False)
    normalizer = edge.flatten(2).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    return (edge / normalizer).clamp(0.0, 5.0)


def _perimeter(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()[None, None]
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (value - eroded).clamp_min(0.0)[0, 0].sum()


def _inner_boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()[None, None]
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (value[0, 0] - eroded[0, 0]) > 0.5


def _ring(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    value = mask.float()[None, None]
    dilated = F.max_pool2d(value, 2 * radius + 1, stride=1, padding=radius)
    return ((dilated[0, 0] > 0.5) & ~mask)


def _attention_pool(values: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    weight = torch.softmax(score.float(), dim=0).to(values.dtype)
    return (weight[:, None] * values).sum(0)


def _quantiles(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(5)
    return torch.quantile(values.float(), values.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90])).to(values.dtype)


def extract_region_records(
    relation: torch.Tensor,
    fused: torch.Tensor,
    raw_sar: torch.Tensor,
    ids: torch.Tensor,
    p_building: torch.Tensor,
    alignment_entropy: torch.Tensor,
    expected_offset: torch.Tensor,
    rgb_edge: torch.Tensor,
    damage: torch.Tensor | None,
    location: torch.Tensor | None,
    names: Sequence[str],
    events: Sequence[str],
    *,
    min_query_building: float = 0.60,
    min_write_building: float = 0.70,
    min_purity: float = 0.80,
    min_tokens: int = 4,
    require_reliable_label: bool = True,
) -> list[RegionRecord]:
    """Create one record per connected SLIC region; never average by class."""
    if relation.ndim != 4 or ids.shape != relation.shape[:1] + relation.shape[-2:]:
        raise ValueError("relation and ids shapes are inconsistent")
    size = relation.shape[-2:]
    p_building = F.interpolate(p_building.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    raw_sar = F.interpolate(raw_sar[:, :1].float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    rgb_edge = F.interpolate(rgb_edge.float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    if damage is not None and location is not None:
        damage_low = F.interpolate(damage[:, None].float(), size=size, mode="nearest")[:, 0].long()
        location_low = F.interpolate(location[:, None].float(), size=size, mode="nearest")[:, 0].long()
    else:
        damage_low = location_low = None
    records: list[RegionRecord] = []
    height, width = size
    yy, xx = torch.meshgrid(
        torch.arange(height, device=relation.device),
        torch.arange(width, device=relation.device), indexing="ij",
    )
    for batch_index in range(relation.shape[0]):
        flat_relation = relation[batch_index].permute(1, 2, 0)
        flat_fused = fused[batch_index].permute(1, 2, 0)
        for region_id in torch.unique(ids[batch_index]).tolist():
            mask = ids[batch_index] == int(region_id)
            area = int(mask.sum().item())
            if area < int(min_tokens):
                continue
            predicted_overlap = float(p_building[batch_index][mask].mean().item())
            if predicted_overlap < float(min_query_building):
                continue
            values = flat_relation[mask]
            score = flat_fused[mask].float().norm(dim=1)
            mean = values.mean(0)
            std = values.float().std(0, unbiased=False).to(values.dtype)
            maximum = values.max(0).values
            attended = _attention_pool(values, score)
            ring = _ring(mask)
            ring_mean = flat_relation[ring].mean(0) if bool(ring.any()) else mean
            core_ring = mean - ring_mean
            perimeter = _perimeter(mask)
            inner_boundary = _inner_boundary(mask)
            compactness = (perimeter.square() / (4.0 * math.pi * max(area, 1))).clamp_max(100.0)
            geometry = relation.new_tensor([
                area / float(height * width),
                float(xx[mask].float().mean().item()) / max(width - 1, 1),
                float(yy[mask].float().mean().item()) / max(height - 1, 1),
                float(compactness.item()),
                predicted_overlap,
                float(alignment_entropy[batch_index, 0][mask].mean().item()),
                float(expected_offset[batch_index, 0][mask].mean().item()),
                float(rgb_edge[batch_index][mask].mean().item()),
                float(rgb_edge[batch_index][inner_boundary].mean().item()) if bool(inner_boundary.any()) else 0.0,
            ])
            descriptor = torch.cat(
                (mean, std, maximum, attended, core_ring, _quantiles(raw_sar[batch_index][mask]), geometry),
                dim=0,
            )
            if damage_low is None or location_low is None:
                soft = descriptor.new_zeros(3)
                label, purity, gt_overlap = 0, 0.0, 0.0
            else:
                valid_building = mask & (location_low[batch_index] > 0) & (damage_low[batch_index] != 255)
                gt_overlap = float(valid_building.sum().item()) / float(area)
                counts = torch.stack([(damage_low[batch_index][valid_building] == c).float().sum() for c in CLASSES])
                soft = counts / counts.sum().clamp_min(1.0)
                purity_tensor, label_index = soft.max(0)
                purity = float(purity_tensor.item())
                label = int(label_index.item()) + 1
            # Training caches contain only reliable labelled regions.  A
            # downstream inference/Oracle audit can explicitly request every
            # label-free P3-building query; this never makes it writable to the
            # Memory and never exposes GT to the proposal generator.
            if require_reliable_label and (
                label == 0 or gt_overlap < min_write_building or purity < min_purity
            ):
                continue
            records.append(RegionRecord(
                descriptor=descriptor,
                soft_label=soft,
                hard_label=label,
                purity=purity,
                building_overlap=gt_overlap,
                area=area,
                event_id=str(events[batch_index]),
                image_id=str(names[batch_index]),
                region_id=int(region_id),
                batch_index=int(batch_index),
            ))
    return records


class MetricKeyEncoder(nn.Module):
    def __init__(self, input_dim: int, key_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden_dim, key_dim),
        )
        self.classifier = nn.Linear(key_dim, 3)

    def forward(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = F.normalize(self.network(descriptor.float()), dim=-1)
        return key, self.classifier(key)


def cross_event_supervised_contrastive(
    key: torch.Tensor,
    label: torch.Tensor,
    event_ids: Sequence[str],
    image_ids: Sequence[str],
    temperature: float = 0.10,
) -> torch.Tensor:
    """Same-class/different-image positives, preferring different events."""
    if key.ndim != 2 or key.shape[0] != label.numel():
        raise ValueError("key/label shape mismatch")
    similarity = key @ key.t() / float(temperature)
    eye = torch.eye(key.shape[0], device=key.device, dtype=torch.bool)
    event_diff = torch.tensor(
        [[event_ids[i] != event_ids[j] for j in range(len(event_ids))] for i in range(len(event_ids))],
        device=key.device, dtype=torch.bool,
    )
    image_diff = torch.tensor(
        [[image_ids[i] != image_ids[j] for j in range(len(image_ids))] for i in range(len(image_ids))],
        device=key.device, dtype=torch.bool,
    )
    same = label[:, None] == label[None, :]
    preferred = same & image_diff & event_diff
    fallback = same & image_diff
    positive = torch.where(preferred.any(1, keepdim=True), preferred, fallback) & ~eye
    valid = positive.any(1)
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    logits = similarity.masked_fill(eye, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probability * positive.float()).sum(1) / positive.sum(1).clamp_min(1)
    return loss[valid].mean()


def retrieve_global_topk(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    topk: int = 8,
    temperature: float = 0.10,
    query_images: Sequence[str] | None = None,
    memory_images: Sequence[str] | None = None,
    query_events: Sequence[str] | None = None,
    memory_events: Sequence[str] | None = None,
    exclude_same_image: bool = True,
    exclude_same_event: bool = False,
) -> dict[str, torch.Tensor]:
    """Global balanced retrieval; no artificial per-class neighbours."""
    query = F.normalize(query.float(), dim=1)
    bank = F.normalize(memory_keys.float(), dim=1)
    similarity = query @ bank.t()

    def provenance_ids(left: Sequence[str], right: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        vocabulary = {value: index for index, value in enumerate(sorted(set(left) | set(right)))}
        left_id = torch.tensor([vocabulary[value] for value in left], device=query.device)
        right_id = torch.tensor([vocabulary[value] for value in right], device=query.device)
        return left_id, right_id

    excluded = torch.zeros_like(similarity, dtype=torch.bool)
    if exclude_same_image and query_images is not None and memory_images is not None:
        query_id, memory_id = provenance_ids(query_images, memory_images)
        excluded |= query_id[:, None] == memory_id[None, :]
    if exclude_same_event and query_events is not None and memory_events is not None:
        query_id, memory_id = provenance_ids(query_events, memory_events)
        excluded |= query_id[:, None] == memory_id[None, :]
    similarity = similarity.masked_fill(excluded, -1e4)
    take = min(int(topk), memory_keys.shape[0])
    score, index = similarity.topk(take, dim=1)
    neighbour_labels = memory_labels[index].long() - 1
    weight = torch.softmax(score / float(temperature), dim=1)
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
        "max_similarity": score[:, 0],
        "indices": index,
        "scores": score,
    }


def safe_region_to_pixel_writeback(
    base_logits: torch.Tensor,
    ids: torch.Tensor,
    region_residual: torch.Tensor,
    region_confidence: torch.Tensor,
    gamma: torch.Tensor,
    boundary_decay: float = 0.25,
) -> torch.Tensor:
    """Preserve Background and vary I/D/X per pixel; never hard-broadcast labels."""
    if base_logits.shape[1] != 4:
        raise ValueError("base logits must contain Background/Intact/Damaged/Destroyed")
    if ids.shape != base_logits.shape[:1] + base_logits.shape[-2:]:
        raise ValueError("ids must match the logit grid")
    output = base_logits.float().clone()
    edge = region_boundary(ids).float()
    for b in range(ids.shape[0]):
        for rid in torch.unique(ids[b]).tolist():
            mask = ids[b] == int(rid)
            index = int(rid)
            if index >= region_residual.shape[1]:
                continue
            confidence = region_confidence[b, index].clamp(0.0, 1.0)
            spatial = (1.0 - edge[b] * (1.0 - float(boundary_decay))) * mask.float()
            delta = torch.tanh(gamma) * confidence * region_residual[b, index, :, None, None]
            output[b, 1:4] += delta * spatial[None]
    # Background logit remains bitwise identical by construction.
    return output.to(base_logits.dtype)
