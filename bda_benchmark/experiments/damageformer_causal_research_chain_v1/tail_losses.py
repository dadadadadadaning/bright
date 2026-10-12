from __future__ import annotations

import torch
import torch.nn.functional as F


def class_counts(labels: torch.Tensor, classes: int = 3) -> torch.Tensor:
    return torch.bincount(labels.long(), minlength=classes).float().clamp_min(1.0)


def balanced_ce(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return F.cross_entropy(logits, labels, weight=weights.to(logits.device))


def logit_adjust_ce(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    prior = counts / counts.sum()
    adjusted = logits + float(tau) * prior.clamp_min(1e-12).log().to(logits.device)
    return F.cross_entropy(adjusted, labels)


def blv_loss(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Building-level adaptation of CVPR'23 BLV Eq.(3).

    Original BLV is pixel-level segmentation.  Here it is deliberately used only
    as a frozen-representation diagnostic baseline on building logits.
    """
    c = torch.log(counts.sum() / counts.clamp_min(1.0))
    scale = c / c.max().clamp_min(1e-12)
    noise = torch.randn_like(logits).mul(float(sigma)).abs().clamp(0.0, 1.0)
    return F.cross_entropy(logits + noise * scale.to(logits.device).view(1, -1), labels)


def seesaw_ce(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor, p: float = 0.8, q: float = 2.0, eps: float = 1e-2) -> torch.Tensor:
    """Classification-only Seesaw CE following the MMDetection class component."""
    num_classes = logits.shape[-1]
    counts = counts.to(logits.device).clamp_min(1.0)
    onehot = F.one_hot(labels, num_classes).float()
    seesaw_weights = torch.ones_like(logits)
    if p > 0:
        ratio = counts[None, :] / counts[:, None]
        idx = (ratio < 1.0).float()
        matrix = ratio.pow(p) * idx + (1.0 - idx)
        seesaw_weights = seesaw_weights * matrix[labels]
    if q > 0:
        scores = F.softmax(logits.detach(), dim=-1)
        self_scores = scores.gather(1, labels[:, None]).clamp_min(eps)
        ratio = scores / self_scores
        idx = (ratio > 1.0).float()
        compensation = ratio.pow(q) * idx + (1.0 - idx)
        seesaw_weights = seesaw_weights * compensation
    adjusted = logits + (seesaw_weights.clamp_min(1e-12).log() * (1.0 - onehot))
    return F.cross_entropy(adjusted, labels)
