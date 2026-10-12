from __future__ import annotations

from dataclasses import dataclass

import torch


def valid_miou(hist: torch.Tensor) -> float:
    """Official BRIGHT foreground (Intact/Damaged/Destroyed) mean IoU."""
    hist = hist.double()
    intersection = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    return float((intersection[1:4] / union[1:4].clamp_min(1.0)).mean().item())


def force_occurrence_predictions(
    logits: torch.Tensor,
    target_affected: bool,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Create an occurrence-only proposal while preserving D-vs-X ordering.

    ``logits`` is [N,4].  Affected proposals raise Damaged and Destroyed by
    the same amount; therefore their relative severity evidence is unchanged.
    Intact proposals raise only Intact.  Background itself is never edited.
    """
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("logits must be [N,4]")
    output = logits.float().clone()
    base = output.argmax(1)
    if target_affected:
        change = base <= 1
        competitor = output[:, :2].max(1).values
        affected = output[:, 2:4].max(1).values
        increment = (competitor - affected + float(epsilon)).clamp_min(0.0)
        output[change, 2:4] += increment[change, None]
    else:
        change = base >= 2
        competitor = torch.stack((output[:, 0], output[:, 2], output[:, 3]), 1).max(1).values
        increment = (competitor - output[:, 1] + float(epsilon)).clamp_min(0.0)
        output[change, 1] += increment[change]
    return output.argmax(1)


def confusion(prediction: torch.Tensor, target: torch.Tensor, classes: int = 4) -> torch.Tensor:
    valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
    values = classes * target[valid].long() + prediction[valid].long()
    return torch.bincount(values, minlength=classes * classes).reshape(classes, classes).double().cpu()


@dataclass
class OracleCandidate:
    index: int
    event: str
    delta: torch.Tensor
    changed: int
    rescued: int
    harmed: int
    affected_recovered: int
    false_affected: int
    individual_gain_pp: float = 0.0
    target_affected: bool | None = None
    source: str = ""


def greedy_metric_oracle(
    base_hist: torch.Tensor,
    candidates: list[OracleCandidate],
) -> tuple[torch.Tensor, list[int], list[OracleCandidate]]:
    """Greedily accept only proposals that improve the current global mIoU.

    This is a reproducible proposal-conditioned upper bound, not an exact
    combinatorial maximum and not a deployable inference rule.
    """
    baseline = valid_miou(base_hist)
    for candidate in candidates:
        candidate.individual_gain_pp = 100.0 * (
            valid_miou(base_hist + candidate.delta) - baseline
        )
    ordered = sorted(candidates, key=lambda row: (-row.individual_gain_pp, row.index))
    current = base_hist.clone().double().cpu()
    accepted: list[int] = []
    for candidate in ordered:
        proposed = current + candidate.delta
        if valid_miou(proposed) > valid_miou(current) + 1e-15:
            current = proposed
            accepted.append(candidate.index)
    return current, accepted, ordered


def greedy_grouped_metric_oracle(
    base_hist: torch.Tensor,
    candidates: list[OracleCandidate],
) -> tuple[torch.Tensor, list[tuple[int, str]], list[OracleCandidate]]:
    """Greedy Oracle with at most one accepted alternative per region index."""
    baseline = valid_miou(base_hist)
    for candidate in candidates:
        candidate.individual_gain_pp = 100.0 * (
            valid_miou(base_hist + candidate.delta) - baseline
        )
    ordered = sorted(candidates, key=lambda row: (-row.individual_gain_pp, row.index, row.source))
    current = base_hist.clone().double().cpu()
    accepted: list[tuple[int, str]] = []
    used_regions: set[int] = set()
    for candidate in ordered:
        if candidate.index in used_regions:
            continue
        proposed = current + candidate.delta
        if valid_miou(proposed) > valid_miou(current) + 1e-15:
            current = proposed
            accepted.append((candidate.index, candidate.source))
            used_regions.add(candidate.index)
    return current, accepted, ordered
