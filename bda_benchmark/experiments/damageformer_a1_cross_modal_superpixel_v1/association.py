"""Memory-conscious local association, pooling and reprojection primitives."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@lru_cache(maxsize=64)
def _candidate_cpu(height: int, width: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    gh = (height + stride - 1) // stride
    gw = (width + stride - 1) // stride
    yy = torch.arange(height, dtype=torch.long)[:, None].expand(height, width)
    xx = torch.arange(width, dtype=torch.long)[None, :].expand(height, width)
    base_y = yy // stride
    base_x = xx // stride
    ids, valid = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy = base_y + dy
            cx = base_x + dx
            ok = (cy >= 0) & (cy < gh) & (cx >= 0) & (cx < gw)
            safe_y = cy.clamp(0, gh - 1)
            safe_x = cx.clamp(0, gw - 1)
            ids.append((safe_y * gw + safe_x).reshape(-1))
            valid.append(ok.reshape(-1))
    return torch.stack(ids), torch.stack(valid), gh, gw


def candidate_map(height: int, width: int, stride: int, device: torch.device):
    ids, valid, gh, gw = _candidate_cpu(height, width, stride)
    return ids.to(device=device), valid.to(device=device), gh, gw


def masked_local_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Convert Bx9xHxW association logits to valid local probabilities."""
    b, nine, h, w = logits.shape
    if nine != 9 or valid.shape != (9, h * w):
        raise ValueError("Expected 9-neighbour logits and matching validity map")
    flat = logits.reshape(b, 9, -1).masked_fill(~valid.unsqueeze(0), -1.0e4)
    return flat.softmax(dim=1)


def soft_pool_unpool(
    content: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
) -> Dict[str, torch.Tensor]:
    """Pool pixels into soft tokens and reproject without a dense KxN matrix."""
    b, c, h, w = content.shape
    n = h * w
    if probabilities.shape != (b, 9, n):
        raise ValueError("Association probability shape does not match content")
    values = content.flatten(2).transpose(1, 2)  # B,N,C
    numerator = content.new_zeros((b, token_count, c))
    mass = content.new_zeros((b, token_count, 1))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1)
        value_idx = idx.expand(b, n, c)
        mass_idx = idx.expand(b, n, 1)
        weight = probabilities[:, j].unsqueeze(-1)
        numerator.scatter_add_(1, value_idx, values * weight)
        mass.scatter_add_(1, mass_idx, weight)
    tokens = numerator / mass.clamp_min(1.0e-6)
    reconstruction = content.new_zeros((b, n, c))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1).expand(b, n, c)
        reconstruction.add_(tokens.gather(1, idx) * probabilities[:, j].unsqueeze(-1))
    return {
        "tokens": tokens,
        "mass": mass.squeeze(-1),
        "reconstruction": reconstruction.transpose(1, 2).reshape(b, c, h, w),
    }


def hard_pool_unpool(content: torch.Tensor, region_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Pool and reproject fixed SLICO regions; labels may have different K per image."""
    b, c, h, w = content.shape
    if region_ids.shape != (b, h, w):
        raise ValueError("Hard region IDs do not match content resolution")
    flat_ids = region_ids.reshape(b, -1).long()
    token_count = int(flat_ids.max().item()) + 1
    values = content.flatten(2).transpose(1, 2)
    numerator = content.new_zeros((b, token_count, c))
    mass = content.new_zeros((b, token_count, 1))
    numerator.scatter_add_(1, flat_ids.unsqueeze(-1).expand_as(values), values)
    mass.scatter_add_(1, flat_ids.unsqueeze(-1), torch.ones_like(flat_ids, dtype=content.dtype).unsqueeze(-1))
    tokens = numerator / mass.clamp_min(1.0)
    reconstructed = tokens.gather(1, flat_ids.unsqueeze(-1).expand(-1, -1, c))
    return {
        "tokens": tokens,
        "mass": mass.squeeze(-1),
        "reconstruction": reconstructed.transpose(1, 2).reshape(b, c, h, w),
    }


def reconstruct_scalar_soft(
    scalar: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    return soft_pool_unpool(scalar, probabilities, candidate_ids, token_count)["reconstruction"]


def normalized_coordinate_grid(batch: int, height: int, width: int, ref: torch.Tensor) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=ref.device, dtype=ref.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=ref.device, dtype=ref.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def fixed_gradient_magnitude(feature: torch.Tensor) -> torch.Tensor:
    """Channel-mean Scharr-like edge magnitude used only as structural evidence."""
    gray = feature.mean(dim=1, keepdim=True)
    kx = gray.new_tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]) / 16.0
    ky = kx.transpose(0, 1)
    gx = F.conv2d(gray, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.view(1, 1, 3, 3), padding=1)
    return (gx.square() + gy.square() + 1.0e-6).sqrt()

