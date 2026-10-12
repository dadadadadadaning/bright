"""Soft same-building association, pooling and evidence-preserving statistics."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_local_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over 9 candidates with per-sample validity masks.

    Args:
        logits: Bx9xHxW
        valid:  Bx9xN bool
    """
    b, nine, h, w = logits.shape
    n = h * w
    if nine != 9 or valid.shape != (b, 9, n):
        raise ValueError(f"expected logits Bx9xHxW and valid Bx9xN, got {logits.shape}, {valid.shape}")
    if bool((~valid.any(dim=1)).any().item()):
        raise ValueError("at least one pixel has no valid region candidate")
    flat = logits.reshape(b, 9, n).masked_fill(~valid, -1.0e4)
    return flat.softmax(dim=1)


def association_entropy(probabilities: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-pixel normalized entropy in [0,1] over valid candidates."""
    p = probabilities.float().clamp_min(1.0e-8)
    ent = -(p * p.log()).sum(dim=1)
    k = valid.sum(dim=1).float().clamp_min(1.0)
    denom = k.log().clamp_min(1.0)
    return (ent / denom).clamp(0.0, 1.0)


def _scatter_moments(
    content: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
) -> Dict[str, torch.Tensor]:
    """Scatter weighted first/second moments with FP32 accumulation.

    Under CUDA AMP/BF16, ``square`` and subsequent arithmetic may be promoted
    differently from the destination tensor used by ``scatter_add_``.  PyTorch
    requires ``self.dtype == src.dtype`` for scatter operations, so accumulating
    moments in explicit FP32 avoids dtype mismatches and is also numerically safer
    for ``E[x^2] - E[x]^2``.  Mean/std are cast back to the feature dtype; mass
    stays FP32 for stable region-area bookkeeping.
    """
    b, c, h, w = content.shape
    n = h * w
    if probabilities.shape != (b, 9, n) or candidate_ids.shape != (b, 9, n):
        raise ValueError("association shape does not match content")

    out_dtype = content.dtype
    values = content.float().flatten(2).transpose(1, 2)  # B,N,C, FP32
    probs = probabilities.float()
    numerator = torch.zeros((b, token_count, c), device=content.device, dtype=torch.float32)
    second = torch.zeros_like(numerator)
    mass = torch.zeros((b, token_count, 1), device=content.device, dtype=torch.float32)

    for j in range(9):
        idx = candidate_ids[:, j].unsqueeze(-1)
        idx_c = idx.expand(-1, -1, c)
        weight = probs[:, j].unsqueeze(-1)
        weighted = values * weight
        weighted_second = values.square() * weight
        numerator.scatter_add_(1, idx_c, weighted)
        second.scatter_add_(1, idx_c, weighted_second)
        mass.scatter_add_(1, idx, weight)

    safe_mass = mass.clamp_min(1.0e-6)
    mean32 = numerator / safe_mass
    variance32 = (second / safe_mass - mean32.square()).clamp_min(0.0)
    std32 = (variance32 + 1.0e-6).sqrt()
    return {
        "mean": mean32.to(out_dtype),
        "std": std32.to(out_dtype),
        "mass": mass.squeeze(-1),
    }


def soft_unpool_tokens(
    tokens: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    b, _, c = tokens.shape
    n = height * width
    reconstruction = tokens.new_zeros((b, n, c))
    probs = probabilities.to(tokens.dtype)
    for j in range(9):
        idx = candidate_ids[:, j].unsqueeze(-1).expand(-1, -1, c)
        src = tokens.gather(1, idx) * probs[:, j].unsqueeze(-1)
        reconstruction.add_(src.to(reconstruction.dtype))
    return reconstruction.transpose(1, 2).reshape(b, c, height, width)


def soft_pool_mean_std(
    content: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
) -> Dict[str, torch.Tensor]:
    result = _scatter_moments(content, probabilities, candidate_ids, token_count)
    h, w = content.shape[-2:]
    result["reconstruction"] = soft_unpool_tokens(
        result["mean"], probabilities, candidate_ids, h, w
    )
    return result


def soft_peak_pool(
    content: torch.Tensor,
    evidence_score: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Differentiable high-evidence pooling inside each soft region.

    ``evidence_score`` is expected to be an unconstrained Bx1xHxW logit.  A
    sigmoid converts it to a bounded strength before the soft exponential.
    """
    b, c, h, w = content.shape
    n = h * w
    out_dtype = content.dtype
    values = content.float().flatten(2).transpose(1, 2)
    score = torch.sigmoid(evidence_score.float()).flatten(2).transpose(1, 2)  # B,N,1
    scale = (score / max(float(temperature), 1.0e-3)).clamp(max=20.0).exp()
    probs = probabilities.float()
    numerator = torch.zeros((b, token_count, c), device=content.device, dtype=torch.float32)
    denom = torch.zeros((b, token_count, 1), device=content.device, dtype=torch.float32)
    for j in range(9):
        idx = candidate_ids[:, j].unsqueeze(-1)
        idx_c = idx.expand(-1, -1, c)
        weight = probs[:, j].unsqueeze(-1) * scale
        numerator.scatter_add_(1, idx_c, values * weight)
        denom.scatter_add_(1, idx, weight)
    return (numerator / denom.clamp_min(1.0e-6)).to(out_dtype)


def pool_scalar_mean(
    scalar: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    if scalar.shape[1] != 1:
        raise ValueError("pool_scalar_mean expects one channel")
    return _scatter_moments(scalar, probabilities, candidate_ids, token_count)["mean"].squeeze(-1)


def normalized_coordinate_grid(batch: int, height: int, width: int, ref: torch.Tensor) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=ref.device, dtype=ref.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=ref.device, dtype=ref.dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((gx, gy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def fixed_gradient_magnitude(feature: torch.Tensor) -> torch.Tensor:
    gray = feature.mean(dim=1, keepdim=True)
    kx = gray.new_tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]) / 16.0
    ky = kx.transpose(0, 1)
    gx = F.conv2d(gray, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.view(1, 1, 3, 3), padding=1)
    return (gx.square() + gy.square() + 1.0e-6).sqrt()


__all__ = [
    "masked_local_softmax",
    "association_entropy",
    "soft_pool_mean_std",
    "soft_peak_pool",
    "soft_unpool_tokens",
    "pool_scalar_mean",
    "normalized_coordinate_grid",
    "fixed_gradient_magnitude",
]
