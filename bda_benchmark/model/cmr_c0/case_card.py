# -*- coding: utf-8 -*-
"""Frozen ``fused_roi`` case cards for the isolated CMR-C0 audit.

CMR-C0 changes only which spatial descriptors are shown to the verifier.  It
does not train a projector, change P3, rebuild the RawMean Top-20 list, or use
ground-truth labels to select query proposals/tokens.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


CASE_CACHE_PROTOCOL = "CMR_C0_ISOLATED_fused_roi_critical_case_cache_v1"
CRITICAL_VARIANTS = (
    "all",
    "di_top10",
    "di_top20",
    "di_top30",
    "di_top20_nms",
)

_DISASTER_BY_INCIDENT = {
    "bata-explosion": "explosion",
    "beirut-explosion": "explosion",
    "haiti-earthquake": "earthquake",
    "morocco-earthquake": "earthquake",
    "noto-earthquake": "earthquake",
    "turkey-earthquake": "earthquake",
    "hawaii-wildfire": "wildfire",
    "marshall-wildfire": "wildfire",
    "congo-volcano": "volcano",
    "la_palma-volcano": "volcano",
    "libya-flood": "flood",
    "mexico-hurricane": "hurricane",
    "myanmar-hurricane": "hurricane",
    "ukraine-conflict": "conflict",
}


def incident_to_disaster_type(incident: str) -> str:
    """Return metadata only; C0 never routes or scores by disaster type."""
    key = str(incident).strip().lower().replace(" ", "_")
    if key in _DISASTER_BY_INCIDENT:
        return _DISASTER_BY_INCIDENT[key]
    suffix = key.rsplit("-", 1)[-1]
    aliases = {
        "earthquake": "earthquake", "wildfire": "wildfire",
        "volcano": "volcano", "flood": "flood", "conflict": "conflict",
        "explosion": "explosion", "hurricane": "hurricane",
    }
    if suffix in aliases:
        return aliases[suffix]
    raise KeyError(f"unknown BRIGHT incident/disaster mapping: {incident!r}")


def region_fingerprint(
    packed_mask: torch.Tensor | np.ndarray,
    bbox: Sequence[int],
    image_shape: Sequence[int],
) -> str:
    """Stable identity fingerprint for one already-frozen predicted component."""
    raw = (
        packed_mask.detach().cpu().contiguous().numpy()
        if isinstance(packed_mask, torch.Tensor)
        else np.ascontiguousarray(packed_mask)
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(tuple(map(int, bbox)), dtype=np.int64).tobytes())
    digest.update(np.asarray(tuple(map(int, image_shape)), dtype=np.int64).tobytes())
    digest.update(raw.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def critical_count(
    token_count: int,
    ratio: float,
    *,
    minimum: int = 4,
    maximum: int = 32,
) -> int:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0,1]")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid critical-token bounds")
    requested = int(math.ceil(float(ratio) * int(token_count)))
    return min(int(token_count), int(maximum), max(int(minimum), requested))


def select_critical_indices(
    di_margin: torch.Tensor,
    xy: torch.Tensor,
    *,
    ratio: float,
    minimum: int = 4,
    maximum: int = 32,
    nms_radius: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Select high D-I positions, optionally enforcing spatial diversity.

    The ordering is deterministic.  NMS first chooses separated points; if an
    extremely small region cannot supply K separated positions, highest-margin
    suppressed points are appended and the number of such backfills is returned.
    """
    margin = di_margin.detach().float().flatten().cpu()
    coordinates = xy.detach().float().cpu()
    if coordinates.shape != (margin.numel(), 2) or margin.numel() == 0:
        raise ValueError("margin/xy must be non-empty and aligned")
    if not torch.isfinite(margin).all() or not torch.isfinite(coordinates).all():
        raise ValueError("margin/xy contain non-finite values")
    take = critical_count(margin.numel(), ratio, minimum=minimum, maximum=maximum)
    order = torch.argsort(margin, descending=True, stable=True).tolist()
    if float(nms_radius) <= 0.0:
        return torch.tensor(order[:take], dtype=torch.long), 0

    selected: list[int] = []
    suppressed: list[int] = []
    radius2 = float(nms_radius) ** 2
    for index in order:
        point = coordinates[index]
        separated = all(
            float((point - coordinates[previous]).square().sum()) > radius2
            for previous in selected
        )
        if separated and len(selected) < take:
            selected.append(index)
        else:
            suppressed.append(index)
    backfill = 0
    if len(selected) < take:
        needed = take - len(selected)
        selected.extend(suppressed[:needed])
        backfill = needed
    return torch.tensor(selected[:take], dtype=torch.long), int(backfill)


def _chw(value: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    tensor = value.detach().float()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("case extraction expects batch size one")
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or (channels is not None and tensor.shape[0] != channels):
        raise ValueError(f"expected CHW tensor with channels={channels}, got {tuple(tensor.shape)}")
    return tensor


def extract_fused_case(
    fused_roi: torch.Tensor,
    p_building: torch.Tensor,
    damage_logits_feature: torch.Tensor,
    component_mask: torch.Tensor | np.ndarray,
    *,
    mask_threshold: float = 0.05,
    minimum: int = 4,
    maximum: int = 32,
    nms_radius: float = 1.5,
) -> Dict[str, object]:
    """Create one label-free 256-D spatial case card from a frozen component."""
    fused = _chw(fused_roi)
    building = _chw(p_building, 1)
    logits = _chw(damage_logits_feature, 4)
    if fused.shape[0] != 256:
        raise RuntimeError(f"CMR-C0 requires fused_roi[256,H,W], got {tuple(fused.shape)}")
    if fused.shape[-2:] != building.shape[-2:] or fused.shape[-2:] != logits.shape[-2:]:
        raise RuntimeError("fused_roi, p_building and damage logits must share a grid")

    component = torch.as_tensor(
        component_mask, dtype=torch.float32, device=fused.device
    )
    if component.ndim != 2 or not bool((component > 0).any()):
        raise ValueError("component_mask must be a non-empty 2-D mask")
    weights = F.interpolate(
        component[None, None], size=fused.shape[-2:], mode="area"
    )[0, 0]
    eligible = weights > float(mask_threshold)
    if not bool(eligible.any()):
        eligible = weights > 0
    if not bool(eligible.any()):
        raise RuntimeError("component vanished on the fused_roi grid")

    xy = torch.nonzero(eligible, as_tuple=False).long()  # deterministic row-major [y,x]
    y, x = xy[:, 0], xy[:, 1]
    features = F.normalize(fused[:, y, x].t(), dim=1, eps=1e-8)
    damage_logits = logits[1:4, y, x].t()  # [Intact, Damaged, Destroyed]
    margin = damage_logits[:, 1] - damage_logits[:, 0]
    height, width = fused.shape[-2:]
    normalizer = torch.tensor(
        [max(height - 1, 1), max(width - 1, 1)],
        dtype=torch.float32,
        device=xy.device,
    )

    # A core pixel has no invalid pixel in its 3x3 neighbourhood.  Explicitly
    # pad the exterior as invalid so image-edge pixels remain boundary pixels.
    # max_pool2d has no separately allocated kernel, therefore CPU/CUDA device
    # mismatch is impossible here.
    invalid = (~eligible).float()[None, None]
    padded_invalid = F.pad(invalid, (1, 1, 1, 1), mode="constant", value=1.0)
    touches_invalid = F.max_pool2d(
        padded_invalid, kernel_size=3, stride=1
    )[0, 0] > 0
    core = eligible & (~touches_invalid)
    is_boundary = (~core[y, x]).bool()

    indices: Dict[str, torch.Tensor] = {
        "all": torch.arange(xy.shape[0], dtype=torch.long),
    }
    nms_backfill: Dict[str, int] = {"all": 0}
    for name, ratio in (("di_top10", 0.10), ("di_top20", 0.20), ("di_top30", 0.30)):
        indices[name], nms_backfill[name] = select_critical_indices(
            margin, xy, ratio=ratio, minimum=minimum, maximum=maximum
        )
    indices["di_top20_nms"], nms_backfill["di_top20_nms"] = select_critical_indices(
        margin, xy, ratio=0.20, minimum=minimum, maximum=maximum,
        nms_radius=nms_radius,
    )
    return {
        "features": features.cpu(),
        "xy": xy.cpu(),
        "normalized_xy": (xy.float() / normalizer).cpu(),
        "damage_logits": damage_logits.cpu(),
        "di_margin": margin.cpu(),
        "building_confidence": building[0, y, x].cpu(),
        "component_weight": weights[y, x].cpu(),
        "is_boundary": is_boundary.cpu(),
        "critical_indices": indices,
        "nms_backfill": nms_backfill,
    }


def case_tokens(payload: Mapping[str, object], row: int, variant: str) -> torch.Tensor:
    if variant not in CRITICAL_VARIANTS:
        raise KeyError(f"unknown critical variant: {variant}")
    features = payload["features"][int(row)].float()
    indices = payload["critical_indices"][variant][int(row)].long()
    selected = features[indices]
    if selected.ndim != 2 or selected.shape[0] == 0:
        raise RuntimeError("critical case contains no tokens")
    return selected


def directional_topk_score(
    query_tokens: torch.Tensor,
    memory_tokens: torch.Tensor,
    *,
    neighbor_k: int = 3,
    query_chunk_size: int = 512,
) -> torch.Tensor:
    """Asymmetric query->memory local retrieval used by frozen CMR-C0."""
    q = F.normalize(query_tokens.float(), dim=1, eps=1e-8)
    m = F.normalize(memory_tokens.float(), dim=1, eps=1e-8)
    if q.ndim != 2 or m.ndim != 2 or q.shape[0] == 0 or m.shape[0] == 0:
        raise ValueError("query and memory token sets must be non-empty matrices")
    if q.shape[1] != m.shape[1]:
        raise ValueError("query and memory feature dimensions differ")
    take = min(max(int(neighbor_k), 1), m.shape[0])
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    totals = q.new_zeros(())
    seen = 0
    for start in range(0, q.shape[0], int(query_chunk_size)):
        similarity = q[start:start + int(query_chunk_size)] @ m.t()
        values = similarity.topk(
            take, dim=1, largest=True, sorted=False
        ).values.mean(dim=1)
        totals = totals + values.sum()
        seen += values.numel()
    return totals / float(seen)


def directional_variant_scores(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    query_indices: Mapping[str, torch.Tensor],
    memory_indices: Mapping[str, torch.Tensor],
    *,
    neighbor_k: int = 3,
    maximum_shared_similarity_elements: int = 2_000_000,
) -> Dict[str, torch.Tensor]:
    """Score all C0 variants while reusing one pairwise similarity when safe."""
    q = F.normalize(query_features.float(), dim=1, eps=1e-8)
    m = F.normalize(memory_features.float(), dim=1, eps=1e-8)
    if q.ndim != 2 or m.ndim != 2 or q.shape[1] != m.shape[1]:
        raise ValueError("full query/memory feature matrices are incompatible")
    if set(query_indices) != set(CRITICAL_VARIANTS) or set(memory_indices) != set(CRITICAL_VARIANTS):
        raise ValueError("variant index mappings are incomplete")
    use_shared = q.shape[0] * m.shape[0] <= int(maximum_shared_similarity_elements)
    shared = q @ m.t() if use_shared else None
    scores: Dict[str, torch.Tensor] = {}
    for variant in CRITICAL_VARIANTS:
        qi = query_indices[variant].long().to(q.device)
        mi = memory_indices[variant].long().to(m.device)
        if qi.numel() == 0 or mi.numel() == 0:
            raise RuntimeError(f"variant {variant} contains an empty token set")
        if shared is None:
            scores[variant] = directional_topk_score(
                q[qi], m[mi], neighbor_k=neighbor_k
            )
            continue
        similarity = shared.index_select(0, qi).index_select(1, mi)
        take = min(max(int(neighbor_k), 1), similarity.shape[1])
        scores[variant] = similarity.topk(
            take, dim=1, largest=True, sorted=False
        ).values.mean()
    return scores


def class_logmeanexp(scores: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    values = scores.float().flatten()
    if values.numel() == 0:
        return values.new_tensor(float("-inf"))
    if temperature <= 0:
        return values.max()
    scaled = values / float(temperature)
    return float(temperature) * (
        torch.logsumexp(scaled, dim=0) - math.log(float(values.numel()))
    )


def validate_case_cache(payload: Mapping[str, object], role: str | None = None) -> None:
    if payload.get("protocol") != CASE_CACHE_PROTOCOL:
        raise RuntimeError("unknown CMR-C0 case-cache protocol")
    if role is not None and payload.get("role") != role:
        raise RuntimeError(f"case-cache role mismatch: {payload.get('role')} != {role}")
    if bool(payload.get("test_accessed", True)):
        raise RuntimeError("CMR-C0 refuses a Test-derived cache")
    rows = torch.as_tensor(payload["source_region_indices"]).long()
    size = rows.numel()
    if size == 0 or torch.unique(rows).numel() != size:
        raise RuntimeError("case-cache source rows must be non-empty and unique")
    list_fields = (
        "features", "xy", "normalized_xy", "damage_logits", "di_margin",
        "building_confidence", "component_weight", "is_boundary",
        "image_ids", "events", "disaster_types", "region_fingerprints",
    )
    for name in list_fields:
        if len(payload[name]) != size:
            raise RuntimeError(f"case-cache field {name} is not row-aligned")
    critical = payload.get("critical_indices", {})
    if set(critical) != set(CRITICAL_VARIANTS):
        raise RuntimeError("case-cache critical variants are incomplete")
    for variant in CRITICAL_VARIANTS:
        if len(critical[variant]) != size:
            raise RuntimeError(f"critical indices for {variant} are not row-aligned")
    for row in range(size):
        count = int(payload["features"][row].shape[0])
        if payload["features"][row].ndim != 2 or payload["features"][row].shape[1] != 256:
            raise RuntimeError("CMR-C0 cache must store [N,256] fused_roi tokens")
        aligned = (
            payload["xy"][row].shape == (count, 2)
            and payload["normalized_xy"][row].shape == (count, 2)
            and payload["damage_logits"][row].shape == (count, 3)
            and payload["di_margin"][row].numel() == count
            and payload["building_confidence"][row].numel() == count
            and payload["component_weight"][row].numel() == count
            and payload["is_boundary"][row].numel() == count
        )
        if not aligned:
            raise RuntimeError(f"case-cache token fields are misaligned at row {row}")
        for variant in CRITICAL_VARIANTS:
            index = critical[variant][row].long()
            if index.numel() == 0 or bool((index < 0).any()) or bool((index >= count).any()):
                raise RuntimeError(f"invalid {variant} critical indices at row {row}")
            if torch.unique(index).numel() != index.numel():
                raise RuntimeError(f"duplicate {variant} critical indices at row {row}")


__all__ = [
    "CASE_CACHE_PROTOCOL", "CRITICAL_VARIANTS", "case_tokens",
    "class_logmeanexp", "critical_count", "directional_topk_score",
    "directional_variant_scores",
    "extract_fused_case", "incident_to_disaster_type", "region_fingerprint",
    "select_critical_indices", "validate_case_cache",
]
