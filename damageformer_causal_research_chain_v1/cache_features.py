from __future__ import annotations

from typing import Dict, Tuple

import torch


def building_feature_from_regions(tokens: torch.Tensor, critical: torch.Tensor) -> torch.Tensor:
    """Label-free 4x token summary: mean, std, top-1, top-2 critical token.

    `critical` must be computed from local cross-modal heterogeneity, not GT.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must be KxD")
    if len(tokens) == 0:
        raise ValueError("building must contain at least one region token")
    critical = critical.reshape(-1)
    if critical.numel() != tokens.shape[0]:
        raise ValueError("critical score size mismatch")
    mean = tokens.mean(0)
    std = tokens.std(0, unbiased=False)
    order = torch.argsort(critical, descending=True)
    top1 = tokens[order[0]]
    top2 = tokens[order[1]] if len(order) > 1 else top1
    return torch.cat((mean, std, top1, top2), dim=0)


def wrong_region_feature(tokens: torch.Tensor, critical: torch.Tensor) -> torch.Tensor:
    """Negative control: retain mean/std but select least critical local regions."""
    if len(tokens) == 0:
        raise ValueError("empty token set")
    mean = tokens.mean(0)
    std = tokens.std(0, unbiased=False)
    order = torch.argsort(critical.reshape(-1), descending=False)
    low1 = tokens[order[0]]
    low2 = tokens[order[1]] if len(order) > 1 else low1
    return torch.cat((mean, std, low1, low2), dim=0)


def region_slice(cache: Dict[str, object], building_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    off = cache["region_offsets"]
    a, b = int(off[building_index]), int(off[building_index + 1])
    return cache["region_tokens"][a:b].float(), cache["region_critical"][a:b].float()
