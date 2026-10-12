# -*- coding: utf-8 -*-
"""Building-level keys from the isolated Stage-2 Occurrence encoder.

O-K0 is a representation audit, not a new classifier.  This module therefore
contains no trainable state.  It pools the already-trained 64-channel S2O
local-token map over an exactly replayed frozen-P3 building component.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def _normalise(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float().flatten()
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("Occurrence key contains non-finite values")
    if float(vector.norm()) <= 1e-12:
        raise RuntimeError("Occurrence key has zero norm")
    return F.normalize(vector, dim=0, eps=1e-8)


def _component_weight(
    component_mask: torch.Tensor,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    mask = component_mask.detach().to(device=device, dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError("component_mask must have shape [H,W]")
    weight = F.interpolate(
        mask[None, None], size=size, mode="area"
    )[0, 0].clamp(0.0, 1.0)
    if float(weight.sum()) <= 1e-6:
        raise RuntimeError("building component vanished at Stage-2 resolution")
    return weight


def masked_mean_key(
    local_tokens: torch.Tensor,
    component_mask: torch.Tensor,
) -> torch.Tensor:
    """L2-normalised area-weighted mean of a [C,h,w] S2O token map."""

    if local_tokens.ndim != 3:
        raise ValueError("local_tokens must have shape [C,h,w]")
    weight = _component_weight(
        component_mask, tuple(local_tokens.shape[-2:]), local_tokens.device
    )
    vector = (local_tokens.float() * weight[None]).sum(dim=(1, 2))
    vector = vector / weight.sum().clamp_min(1e-6)
    return _normalise(vector)


def soft_occurrence_key(
    local_tokens: torch.Tensor,
    occurrence_response: torch.Tensor,
    component_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Pool tokens with positive Occurrence-response attention.

    The fractional area weight is included inside the softmax.  Consequently
    boundary cells with only tiny component overlap cannot dominate merely
    because they have a large response.  No label is used in this operation.
    """

    if local_tokens.ndim != 3:
        raise ValueError("local_tokens must have shape [C,h,w]")
    if occurrence_response.ndim == 3 and occurrence_response.shape[0] == 1:
        occurrence_response = occurrence_response[0]
    if occurrence_response.ndim != 2:
        raise ValueError("occurrence_response must have shape [h,w] or [1,h,w]")
    if tuple(occurrence_response.shape) != tuple(local_tokens.shape[-2:]):
        raise ValueError("Occurrence response and local tokens are not aligned")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    weight = _component_weight(
        component_mask, tuple(local_tokens.shape[-2:]), local_tokens.device
    )
    valid = weight > 0.0
    logits = occurrence_response.float() / float(temperature)
    logits = logits + weight.clamp_min(1e-12).log()
    logits = logits.masked_fill(~valid, float("-inf"))
    attention = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
    if not bool(torch.isfinite(attention).all()):
        raise FloatingPointError("soft Occurrence attention is non-finite")
    vector = (local_tokens.float() * attention[None]).sum(dim=(1, 2))
    return _normalise(vector)


def _temperature_suffix(temperature: float) -> str:
    value = f"{float(temperature):g}".replace("-", "m").replace(".", "p")
    return f"s2o_soft_t{value}"


def occurrence_key_variant_names(temperatures: Sequence[float]) -> tuple[str, ...]:
    names = ["s2o_masked_mean"]
    for temperature in temperatures:
        name = _temperature_suffix(float(temperature))
        if name in names:
            raise ValueError("duplicate Occurrence temperature")
        names.append(name)
    return tuple(names)


def extract_occurrence_key_variants(
    auxiliary: Mapping[str, torch.Tensor],
    component_mask: torch.Tensor,
    *,
    batch_index: int = 0,
    temperatures: Sequence[float] = (0.25, 0.5, 1.0),
) -> dict[str, torch.Tensor]:
    """Extract every predeclared O-K0 key for one building component."""

    tokens = auxiliary["stage2_local_tokens"]
    response = auxiliary["stage2_raw_delta"]
    if tokens.ndim != 4 or response.ndim != 4:
        raise ValueError("S2O auxiliary maps must have shape [B,C,h,w]")
    if not 0 <= int(batch_index) < tokens.shape[0]:
        raise IndexError("batch_index is outside the S2O batch")
    token_map = tokens[int(batch_index)]
    response_map = response[int(batch_index)]
    result = {
        "s2o_masked_mean": masked_mean_key(token_map, component_mask),
    }
    for temperature in temperatures:
        result[_temperature_suffix(float(temperature))] = soft_occurrence_key(
            token_map,
            response_map,
            component_mask,
            temperature=float(temperature),
        )
    expected = occurrence_key_variant_names(temperatures)
    if tuple(result) != expected:
        raise AssertionError("Occurrence key variant contract changed")
    return {name: value.detach().cpu() for name, value in result.items()}


__all__ = [
    "occurrence_key_variant_names",
    "extract_occurrence_key_variants",
    "masked_mean_key",
    "soft_occurrence_key",
]
