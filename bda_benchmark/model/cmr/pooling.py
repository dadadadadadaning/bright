# -*- coding: utf-8 -*-
"""Frozen-P3 building representations for the CMR pooling audit.

The accepted P3 head already computes global, critical, top-k, attention-bag,
and penultimate occurrence representations.  The historical RM0 cache kept
only a mean-pooled ``fused_roi`` vector.  This module exposes the alternatives
without changing the P3 forward contract or introducing trainable parameters.
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


CMR_KEY_VARIANTS = (
    "mean",
    "mean_std",
    "hard_critical",
    "topk_mean",
    "soft_topk",
    "attention_bag",
    "exist_embedding",
    "global_exist",
)


def _normalise(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float().flatten()
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("CMR pooling produced a non-finite key")
    if float(vector.norm()) <= 1e-12:
        raise RuntimeError("CMR pooling produced a zero-norm key")
    return F.normalize(vector, dim=0)


def _resize_mask(mask: torch.Tensor, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    tensor = mask.detach().to(device=device, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"component mask must be [H,W], got {tuple(tensor.shape)}")
    return F.interpolate(tensor[None, None], size=size, mode="area")[0, 0].clamp(0, 1)


def _weighted_mean_std(feature: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if feature.ndim != 3 or weight.ndim != 2 or feature.shape[-2:] != weight.shape:
        raise ValueError("feature/weight must be [C,H,W] and [H,W]")
    denominator = weight.sum().clamp_min(1e-6)
    mean = (feature * weight[None]).sum(dim=(1, 2)) / denominator
    variance = (
        (feature - mean[:, None, None]).square() * weight[None]
    ).sum(dim=(1, 2)) / denominator
    return mean, variance.clamp_min(0).sqrt()


def build_cmr_feature_maps(
    head: torch.nn.Module,
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    fused_feature: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run the frozen MICB projections once for a batch.

    Inputs follow the existing P3 auxiliary contract and must share
    ``[B,256,h,w]``.  No labels or region masks are used here.
    """

    if not (rgb_feature.shape == sar_feature.shape == fused_feature.shape):
        raise ValueError("RGB, SAR, and fused features must share [B,C,H,W]")
    rgb = head.rgb_proj(rgb_feature)
    sar = head.sar_proj(sar_feature)
    fused = head.fused_proj(fused_feature)
    token_map = head.cross_token(
        torch.cat((rgb, sar, torch.abs(rgb - sar), rgb * sar, fused), dim=1)
    )
    return {
        "token_map": token_map,
        "change_energy": torch.abs(rgb - sar).mean(dim=1),
        "fused_feature": fused_feature,
    }


def _select_tokens(
    token_map: torch.Tensor,
    weights: torch.Tensor,
    change_energy: torch.Tensor,
    *,
    mask_threshold: float,
    max_tokens: int,
) -> torch.Tensor:
    indices = torch.nonzero(weights > float(mask_threshold), as_tuple=False)
    if indices.shape[0] == 0:
        indices = torch.nonzero(weights > 0, as_tuple=False)
    if indices.shape[0] == 0:
        raise RuntimeError("a non-empty component vanished at feature resolution")
    flat = indices[:, 0] * weights.shape[1] + indices[:, 1]
    tokens = token_map.permute(1, 2, 0).reshape(-1, token_map.shape[0])[flat]
    energy = change_energy.reshape(-1)[flat]
    if tokens.shape[0] <= int(max_tokens):
        return tokens

    # Match P3's deterministic half-high-change / half-spatially-spread policy.
    high_count = int(max_tokens) // 2
    high = torch.topk(energy, k=high_count, largest=True, sorted=False).indices
    spread_count = int(max_tokens) - high_count
    spread = torch.linspace(
        0, tokens.shape[0] - 1, steps=spread_count, device=tokens.device
    ).round().long()
    chosen = torch.unique(torch.cat((high, spread), dim=0), sorted=False)
    if chosen.numel() < int(max_tokens):
        remaining = torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.bool)
        remaining[chosen] = False
        fill = torch.nonzero(remaining, as_tuple=False)[: int(max_tokens) - chosen.numel(), 0]
        chosen = torch.cat((chosen, fill), dim=0)
    return tokens[chosen[: int(max_tokens)]]


def extract_cmr_pooling_keys(
    head: torch.nn.Module,
    feature_maps: Mapping[str, torch.Tensor],
    component_mask: torch.Tensor,
    *,
    batch_index: int = 0,
) -> Dict[str, torch.Tensor]:
    """Return all controlled pooling variants for one predicted building.

    Every returned vector is finite and L2-normalised.  ``mean`` exactly uses
    the historical RM0 ``fused_roi`` representation.  The other keys reuse
    frozen P3/MICB parameters and add no trainable state.
    """

    token_batch = feature_maps["token_map"]
    fused_batch = feature_maps["fused_feature"]
    energy_batch = feature_maps["change_energy"]
    if not 0 <= int(batch_index) < token_batch.shape[0]:
        raise IndexError("batch_index is outside the feature batch")
    token_map = token_batch[int(batch_index)]
    fused = fused_batch[int(batch_index)]
    energy = energy_batch[int(batch_index)]
    weights = _resize_mask(component_mask, token_map.shape[-2:], token_map.device)
    if float(weights.sum()) <= 1e-6:
        raise RuntimeError("component has no support at MICB feature resolution")

    fused_weights = _resize_mask(component_mask, fused.shape[-2:], fused.device)
    fused_mean, fused_std = _weighted_mean_std(fused, fused_weights)
    global_token, _ = _weighted_mean_std(token_map, weights)
    tokens = _select_tokens(
        token_map,
        weights,
        energy,
        mask_threshold=float(head.mask_threshold),
        max_tokens=int(head.max_tokens),
    )

    local_logits = head.local_exist(tokens)[:, 0]
    critical_index = local_logits.argmax()
    hard_critical = tokens[critical_index]
    local_evidence = local_logits.max()[None]
    count = min(int(head.topk), tokens.shape[0])
    top_result = torch.topk(local_logits, k=count, largest=True, sorted=True)
    top_tokens = tokens[top_result.indices]
    topk_mean = top_tokens.mean(dim=0)
    top_weights = torch.softmax(
        top_result.values / float(head.attention_temperature), dim=0
    )
    soft_topk = (top_weights[:, None] * top_tokens).sum(dim=0)

    queries = F.normalize(head.query(tokens), dim=1)
    critical_query = F.normalize(head.query(hard_critical[None]), dim=1)[0]
    attention = torch.softmax(
        (queries @ critical_query) / float(head.attention_temperature), dim=0
    )
    attention_bag = (attention[:, None] * head.value(tokens)).sum(dim=0)

    exist_input = torch.cat(
        (global_token, hard_critical, topk_mean, attention_bag, local_evidence),
        dim=0,
    )
    # Reuse P3's trained occurrence representation immediately before logits.
    exist_embedding = head.exist_head[:-1](exist_input[None])[0]

    result = {
        "mean": _normalise(fused_mean),
        "mean_std": _normalise(torch.cat((fused_mean, fused_std), dim=0)),
        "hard_critical": _normalise(hard_critical),
        "topk_mean": _normalise(topk_mean),
        "soft_topk": _normalise(soft_topk),
        "attention_bag": _normalise(attention_bag),
        "exist_embedding": _normalise(exist_embedding),
        "global_exist": _normalise(torch.cat((global_token, exist_embedding), dim=0)),
    }
    if tuple(result) != CMR_KEY_VARIANTS:
        raise AssertionError("CMR pooling key contract changed unexpectedly")
    return {name: value.detach().cpu() for name, value in result.items()}


def extract_cmr_local_tokens(
    head: torch.nn.Module,
    feature_maps: Mapping[str, torch.Tensor],
    component_mask: torch.Tensor,
    *,
    batch_index: int = 0,
) -> torch.Tensor:
    """Return the exact frozen MICB local tokens selected inside one region.

    This is the local evidence used by CMR-V0.  It deliberately reuses P3's
    deterministic token policy and adds no trainable descriptor.
    """

    token_batch = feature_maps["token_map"]
    energy_batch = feature_maps["change_energy"]
    if not 0 <= int(batch_index) < token_batch.shape[0]:
        raise IndexError("batch_index is outside the feature batch")
    token_map = token_batch[int(batch_index)]
    energy = energy_batch[int(batch_index)]
    weights = _resize_mask(component_mask, token_map.shape[-2:], token_map.device)
    tokens = _select_tokens(
        token_map,
        weights,
        energy,
        mask_threshold=float(head.mask_threshold),
        max_tokens=int(head.max_tokens),
    )
    tokens = F.normalize(tokens.float(), dim=1, eps=1e-8)
    if not bool(torch.isfinite(tokens).all()) or tokens.shape[0] == 0:
        raise RuntimeError("CMR local token extraction produced invalid tokens")
    return tokens.detach().cpu()
