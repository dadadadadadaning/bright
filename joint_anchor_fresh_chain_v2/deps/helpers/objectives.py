"""A single-term Lovasz ablation; all four outputs and all valid pixels remain.

The control matches BRIGHT region_nll + lovasz_softmax(classes='present').
The candidate omits class 0's Lovasz numerator, retaining the ORIGINAL number
of present classes as denominator. This is not classes=[1, 2, 3], and is not
foreground-pixel filtering. NLL remains an unweighted mean of all valid pixels.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F

IGNORE_INDEX = 255
CLASS_COUNT = 4


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    # Match the inherited implementation, including FP32 cumulative sums.
    count = len(gt_sorted)
    if count == 0:
        return gt_sorted
    total = gt_sorted.sum()
    intersection = total - gt_sorted.float().cumsum(0)
    union = total + (1.0 - gt_sorted).float().cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-6)
    if count > 1:
        gradient[1:count] = gradient[1:count] - gradient[:-1].clone()
    return gradient


def objective_terms(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | torch.Tensor | None = None,
    verify_reference: bool = False,
) -> dict[str, Any]:
    """Return differentiable scalar objectives and per-class contributions.

    Predictions are B,4,H,W; labels are B,H,W with classes 0..3 or ignore 255.
    ``class_*_contribution`` sum to their corresponding complete loss (within
    floating point rounding); they are not conditional per-class means.

    A callable reference is evaluated under no_grad only when verification is
    requested. A scalar reference can instead carry the exact inherited graph
    for control training; that tensor is returned as ``loss``. The candidate
    never derives from subtracting a reference scalar, so the removed term
    cannot accidentally retain a gradient.
    """
    if probability.ndim != 4 or probability.shape[1] != CLASS_COUNT:
        raise ValueError('Expected B,4,H,W class probabilities')
    if target.shape != (probability.shape[0], *probability.shape[2:]):
        raise ValueError('Expected B,H,W labels matching probabilities')
    if target.device != probability.device:
        raise ValueError('Predictions and labels must share a device')
    if not probability.is_floating_point():
        raise ValueError('Class probabilities must be floating point')
    if target.is_floating_point() and not bool(torch.isfinite(target).all() & (target == target.long()).all()):
        raise ValueError('Labels must be finite integers 0..3 or ignore 255')
    labels = target.long().reshape(-1).clone()
    valid = labels != IGNORE_INDEX
    if bool(((labels[valid] < 0) | (labels[valid] >= CLASS_COUNT)).any()):
        raise ValueError('Target class outside 0..3 (ignore=255)')
    selected_labels = labels[valid].contiguous()
    valid_count = int(valid.sum().item())
    counts = torch.bincount(selected_labels, minlength=CLASS_COUNT)
    present_ids = [cls for cls, count in enumerate(counts.detach().cpu().tolist()) if count]
    present_count = len(present_ids)

    # Preserve original precision: only NLL's table is explicitly cast to FP32.
    table = probability.permute(0, 2, 3, 1).contiguous().view(-1, CLASS_COUNT)
    selected = table[valid]
    nll_table = probability.float().permute(0, 2, 3, 1).reshape(-1, CLASS_COUNT).contiguous()
    zero = nll_table.sum() * 0.0
    nll_by_class = [zero for _ in range(CLASS_COUNT)]
    lovasz_raw = [zero for _ in range(CLASS_COUNT)]
    if valid_count:
        logp = nll_table[valid].clamp_min(1e-8).log().contiguous()
        nll = F.nll_loss(logp, selected_labels)
        per_pixel_nll = -logp.gather(1, selected_labels[:, None]).squeeze(1)
        for cls in present_ids:
            nll_by_class[cls] = per_pixel_nll[selected_labels == cls].sum() / valid_count
            # Keep every valid background/foreground pixel as a possible negative.
            foreground = (selected_labels == cls).float()
            errors = (foreground - selected[:, cls]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            lovasz_raw[cls] = torch.dot(sorted_errors, _lovasz_grad(foreground[permutation]))
        # Exact original reduction order: stack ONLY present terms then mean.
        lovasz = torch.stack([lovasz_raw[cls] for cls in present_ids]).mean()
        lovasz_contributions = torch.stack(lovasz_raw) / present_count
        foreground_terms = [lovasz_raw[cls] for cls in present_ids if cls != 0]
        candidate_lovasz = (torch.stack(foreground_terms).sum() / present_count) if foreground_terms else zero
    else:
        nll = zero
        lovasz = zero
        candidate_lovasz = zero
        lovasz_contributions = torch.stack(lovasz_raw)

    computed_original = nll + lovasz
    original = reference_loss if torch.is_tensor(reference_loss) else computed_original
    if torch.is_tensor(reference_loss) and (reference_loss.ndim != 0 or reference_loss.device != probability.device):
        raise ValueError('Reference loss must be a scalar on the prediction device')
    if verify_reference:
        if reference_loss is None:
            raise ValueError('verify_reference requires the inherited loss callable or scalar')
        with torch.no_grad():
            expected = reference_loss(probability.detach(), target.detach()) if callable(reference_loss) else reference_loss.detach()
        if not torch.is_tensor(expected) or expected.ndim != 0:
            raise ValueError('Inherited reference loss must return a scalar tensor')
        if not bool(torch.isfinite(expected) & torch.isfinite(computed_original.detach())):
            raise RuntimeError('Non-finite inherited/objective loss during reference verification')
        if not torch.allclose(computed_original.detach(), expected.detach(), rtol=2e-6, atol=2e-7):
            raise RuntimeError('Original objective differs from inherited runtime.loss: '
                               f'computed={float(computed_original.detach())}, reference={float(expected.detach())}')

    return {
        'loss': original,
        'candidate_loss': nll + candidate_lovasz,
        'nll': nll,
        'lovasz': lovasz,
        'candidate_lovasz': candidate_lovasz,
        'background_lovasz_contribution': lovasz_contributions[0],
        'class_nll_contribution': torch.stack(nll_by_class),
        'class_lovasz_contribution': lovasz_contributions,
        'class_valid_counts': counts,
        'valid_count': valid_count,
        'present_count': present_count,
    }


def serialized_terms(terms: dict[str, Any]) -> dict[str, Any]:
    """Detach report values without retaining graphs or emitting non-JSON types."""
    result: dict[str, Any] = {}
    for name, value in terms.items():
        if torch.is_tensor(value):
            result[name] = value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
        elif isinstance(value, (int, float, bool, str)) or value is None:
            result[name] = value
        else:
            raise TypeError(f'Unsupported report type for {name}: {type(value).__name__}')
    return result
