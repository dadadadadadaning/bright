from __future__ import annotations

from math import ceil, sqrt
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def hard_region_ids(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert the local soft assignment to one deterministic token ID per pixel."""
    probabilities = aux["association_probabilities"]
    candidates = aux["candidate_ids"]
    slot = probabilities.argmax(dim=1, keepdim=True)
    return candidates.gather(1, slot).squeeze(1).reshape(aux["building_ids"].shape)


def resize_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(labels[:, None].float(), size=size, mode="nearest")[:, 0].long()


def _moments(value: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = value[:, mask].float()
    if selected.shape[1] == 0:
        z = value.new_zeros((value.shape[0],), dtype=torch.float32)
        return z, z, z
    return selected.mean(1), selected.std(1, unbiased=False), selected.max(1).values


def region_descriptor(
    model,
    aux: Dict[str, torch.Tensor],
    batch_index: int,
    mask: torch.Tensor,
    building_area: int,
) -> torch.Tensor:
    """Use identical pooling and the same frozen token encoder for real/grid regions."""
    rgb = aux["rgb_alignment_value"][batch_index]
    sar = aux["aligned_sar_value"][batch_index]
    relation = aux["region_relation"][batch_index]
    confidence = aux["region_confidence"][batch_index]
    entropy = aux["region_alignment_entropy"][batch_index]
    rgb_mean, _, _ = _moments(rgb, mask)
    sar_mean, _, _ = _moments(sar, mask)
    rel_mean, rel_std, rel_peak = _moments(relation, mask)
    conf_mean, _, _ = _moments(confidence, mask)
    ent_mean, _, _ = _moments(entropy, mask)
    area_ratio = relation.new_tensor([float(mask.sum()) / max(1.0, float(building_area))]).float()
    token_input = torch.cat(
        (rgb_mean, sar_mean, rel_mean, rel_std, rel_peak, conf_mean, ent_mean, area_ratio), dim=0
    )[None].to(next(model.biscp.token_encoder.parameters()).dtype)
    return model.biscp.token_encoder(token_input).float()[0]


def a1_descriptor(aux: Dict[str, torch.Tensor], batch_index: int, mask: torch.Tensor) -> torch.Tensor:
    probability = aux["parent_damage_probability"][batch_index].float()
    p_mean, p_std, p_max = _moments(probability, mask)
    entropy_map = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(0, keepdim=True)
    margin_map = (probability[1:2] - probability[2:3]).abs()
    entropy = torch.cat(_moments(entropy_map, mask))
    margin = torch.cat(_moments(margin_map, mask))
    # Exactly 18 dimensions; all probe variants receive the same final layout.
    return torch.cat((p_mean, p_std, p_max, entropy, margin), dim=0).float()


def matched_grid_mask(mask: torch.Tensor, building_mask: torch.Tensor) -> torch.Tensor:
    """Square local window with approximately the same area, clipped to one building."""
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return mask.clone()
    cy, cx = coords.float().mean(0)
    side = max(1, int(ceil(sqrt(float(mask.sum())))))
    half = side // 2
    h, w = mask.shape
    y0 = max(0, min(h - side, int(round(float(cy))) - half))
    x0 = max(0, min(w - side, int(round(float(cx))) - half))
    output = torch.zeros_like(mask)
    output[y0 : min(h, y0 + side), x0 : min(w, x0 + side)] = True
    output &= building_mask
    return output if bool(output.any()) else mask.clone()


def semantic_edges(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(valid)
    dv = valid[1:] & valid[:-1] & (labels[1:] != labels[:-1])
    dh = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    edge[1:] |= dv
    edge[:-1] |= dv
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    return edge


def partition_edges(region_ids: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(building)
    # Include both internal region boundaries and predicted building outlines.
    # Requiring at least one endpoint to be building suppresses background-only noise.
    dv = (building[1:] | building[:-1]) & (region_ids[1:] != region_ids[:-1])
    dh = (building[:, 1:] | building[:, :-1]) & (region_ids[:, 1:] != region_ids[:, :-1])
    edge[1:] |= dv
    edge[:-1] |= dv
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    return edge
