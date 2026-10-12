from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    metrics_from_histogram,
)


VARIANTS = (
    "a0_raw_p3",
    "a1_majority",
    "a2_mean_probability",
    "a3_micb",
    "a4_gated_majority",
    "a5_gated_core_majority",
)


@dataclass(frozen=True)
class RegionEvidence:
    label: int
    area: int
    foreground_pixels: int
    purity: float
    margin: float


def region_evidence(
    pixel_prediction: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> RegionEvidence:
    """Prediction-only evidence used by the confidence gate.

    Purity uses the full localization region as denominator. Background pixels
    inside a predicted building therefore reduce confidence instead of being
    silently discarded. The chosen label remains foreground-only.
    """

    if pixel_prediction.shape != mask.shape:
        raise ValueError("pixel_prediction and mask have different shapes")
    area = int(mask.sum().item())
    if area < 1:
        raise ValueError("cannot score an empty region")
    values = pixel_prediction[mask].long()
    counts = torch.bincount(values, minlength=4)[1:4]
    label = foreground_majority_label(pixel_prediction, probabilities, mask)
    winning_votes = int(counts[label - 1].item())
    scores = probabilities[1:4, mask].mean(dim=1)
    alternatives = torch.cat((scores[: label - 1], scores[label:]))
    margin = float((scores[label - 1] - alternatives.max()).item())
    return RegionEvidence(
        label=label,
        area=area,
        foreground_pixels=int(counts.sum().item()),
        purity=float(winning_votes / area),
        margin=margin,
    )


def gate_region(
    evidence: RegionEvidence,
    *,
    minimum_purity: float,
    minimum_margin: float,
    maximum_area: int = 0,
) -> bool:
    if not 0.0 <= minimum_purity <= 1.0:
        raise ValueError("minimum_purity must be in [0,1]")
    if minimum_margin < 0.0:
        raise ValueError("minimum_margin must be nonnegative")
    return bool(
        evidence.purity >= minimum_purity
        and evidence.margin >= minimum_margin
        and (maximum_area <= 0 or evidence.area <= maximum_area)
    )


def erode_region(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Return a square-kernel interior; radius=0 preserves the full region."""

    if mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return mask.clone().bool()
    size = 2 * radius + 1
    values = mask.float()[None, None]
    summed = F.conv2d(values, torch.ones((1, 1, size, size)), padding=radius)
    return (summed[0, 0] == float(size * size)) & mask.bool()


def gated_majority_feedback(
    base_prediction: torch.Tensor,
    probabilities: torch.Tensor,
    regions: Sequence[torch.Tensor],
    *,
    minimum_purity: float,
    minimum_margin: float,
    maximum_area: int = 0,
    core_radius: int = 0,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Apply majority only to prediction-confident regions and return evidence."""

    output = base_prediction.clone()
    records: list[dict[str, object]] = []
    for index, mask in enumerate(regions):
        evidence = region_evidence(base_prediction, probabilities, mask)
        accepted = gate_region(
            evidence,
            minimum_purity=minimum_purity,
            minimum_margin=minimum_margin,
            maximum_area=maximum_area,
        )
        write_mask = erode_region(mask, core_radius) if accepted else torch.zeros_like(mask)
        core_area = int(write_mask.sum().item())
        applied = bool(accepted and core_area > 0)
        if applied:
            output[write_mask] = evidence.label
        records.append(
            {
                "region_index": index,
                "label": evidence.label,
                "area": evidence.area,
                "foreground_pixels": evidence.foreground_pixels,
                "purity": evidence.purity,
                "margin": evidence.margin,
                "purity_pass": evidence.purity >= minimum_purity,
                "margin_pass": evidence.margin >= minimum_margin,
                "area_pass": maximum_area <= 0 or evidence.area <= maximum_area,
                "gate_accepted": accepted,
                "applied": applied,
                "write_area": core_area,
            }
        )
    return output, records


def predicted_building_regions(
    localization_prediction: torch.Tensor,
    minimum_area: int = 32,
) -> list[torch.Tensor]:
    """Build regions only from model localization; semantic GT is never read here."""

    if localization_prediction.ndim != 2:
        raise ValueError("localization_prediction must be [H,W]")
    output: list[torch.Tensor] = []
    for component in connected_components_4(
        localization_prediction.detach().cpu().numpy() > 0,
        minimum_area=int(minimum_area),
    ):
        output.append(torch.from_numpy(component.copy()).bool())
    return output


def foreground_majority_label(
    pixel_prediction: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    """Foreground-only majority; mean foreground probability breaks empty/tied votes."""

    values = pixel_prediction[mask].long()
    counts = torch.bincount(values, minlength=4)[1:4]
    maximum = int(counts.max().item()) if counts.numel() else 0
    winners = torch.nonzero(counts == maximum, as_tuple=False).flatten()
    if maximum > 0 and winners.numel() == 1:
        return int(winners[0].item()) + 1
    return mean_probability_label(probabilities, mask)


def mean_probability_label(probabilities: torch.Tensor, mask: torch.Tensor) -> int:
    if probabilities.ndim != 3 or int(probabilities.shape[0]) != 4:
        raise ValueError("probabilities must be [4,H,W]")
    if mask.shape != probabilities.shape[-2:]:
        raise ValueError("mask and probabilities have different spatial shapes")
    if not bool(mask.any().item()):
        raise ValueError("cannot pool an empty region")
    scores = probabilities[1:4, mask].mean(dim=1)
    return int(scores.argmax().item()) + 1


def broadcast_labels(
    base_prediction: torch.Tensor,
    regions: Sequence[torch.Tensor],
    labels: Sequence[int],
) -> torch.Tensor:
    if len(regions) != len(labels):
        raise ValueError("regions and labels must have the same length")
    output = base_prediction.clone()
    for mask, label in zip(regions, labels):
        if int(label) not in (1, 2, 3):
            raise ValueError(f"region label must be 1/2/3, got {label}")
        output[mask] = int(label)
    return output


def stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    variants: Sequence[str] = VARIANTS,
    baseline: str = "a0_raw_p3",
    iterations: int = 10_000,
    seed: int = 3407,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Paired image bootstrap, resampled independently inside every event."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    by_event: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_event[str(row["event"])].append(row)
    rng = np.random.default_rng(seed)
    deltas = {variant: [] for variant in variants if variant != baseline}
    pairs = list(comparison_pairs or [])
    for left, right in pairs:
        if left not in variants or right not in variants:
            raise ValueError(f"unknown bootstrap comparison: {left} vs {right}")
    pair_deltas = {(left, right): [] for left, right in pairs}
    for _ in range(iterations):
        sampled = {variant: torch.zeros((4, 4), dtype=torch.float64) for variant in variants}
        for event_rows in by_event.values():
            indices = rng.integers(0, len(event_rows), size=len(event_rows))
            for index in indices:
                histograms = event_rows[int(index)]["histograms"]
                for variant in variants:
                    sampled[variant] += torch.as_tensor(histograms[variant], dtype=torch.float64)
        baseline_miou = metrics_from_histogram(sampled[baseline])["valid_miou"]
        for variant in deltas:
            value = metrics_from_histogram(sampled[variant])["valid_miou"]
            deltas[variant].append(100.0 * (value - baseline_miou))
        sampled_miou = {
            variant: metrics_from_histogram(sampled[variant])["valid_miou"]
            for variant in variants
        }
        for left, right in pairs:
            pair_deltas[(left, right)].append(
                100.0 * (sampled_miou[left] - sampled_miou[right])
            )
    output: dict[str, object] = {
        "iterations": int(iterations),
        "seed": int(seed),
        "stratification": "event",
        "unit": "percentage_points",
        "deltas_vs_a0": {},
    }
    for variant, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        output["deltas_vs_a0"][variant] = {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    output["comparisons"] = {}
    for (left, right), values in pair_deltas.items():
        array = np.asarray(values, dtype=np.float64)
        output["comparisons"][f"{left}_minus_{right}"] = {
            "left": left,
            "right": right,
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    return output


def prediction_change_audit(
    base_prediction: torch.Tensor,
    updated_prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Count beneficial and harmful changes; target is used for audit only."""

    if not (
        base_prediction.shape == updated_prediction.shape == target.shape
    ):
        raise ValueError("base, updated, and target must have identical shapes")
    selected = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.bool()
    selected &= (target >= 0) & (target <= 3)
    changed = selected & (base_prediction != updated_prediction)
    base_correct = base_prediction == target
    updated_correct = updated_prediction == target
    corrected = changed & ~base_correct & updated_correct
    harmed = changed & base_correct & ~updated_correct
    transition = torch.zeros((4, 4), dtype=torch.long)
    if bool(changed.any().item()):
        encoded = 4 * base_prediction[changed].long() + updated_prediction[changed].long()
        transition = torch.bincount(encoded, minlength=16).reshape(4, 4).cpu()
    corrected_count = int(corrected.sum().item())
    harmed_count = int(harmed.sum().item())
    return {
        "selected_pixels": int(selected.sum().item()),
        "changed_pixels": int(changed.sum().item()),
        "corrected_pixels": corrected_count,
        "harmed_pixels": harmed_count,
        "net_gain_pixels": corrected_count - harmed_count,
        "raw_to_updated_transition": transition.tolist(),
    }


def constant_label_region_audit(
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    region_mask: torch.Tensor,
    write_mask: torch.Tensor,
    label: int,
) -> dict[str, object]:
    """Efficient audit for writing one label inside one region.

    This avoids cloning an entire image for every connected component.
    """

    if not (
        base_prediction.shape
        == target.shape
        == region_mask.shape
        == write_mask.shape
    ):
        raise ValueError("all region-audit tensors must have identical shapes")
    if int(label) not in (1, 2, 3):
        raise ValueError("label must be 1/2/3")
    valid_target = (target >= 0) & (target <= 3)
    region_mask = region_mask.bool() & valid_target
    write_mask = write_mask.bool() & region_mask
    changed = write_mask & (base_prediction != int(label))
    corrected = changed & (base_prediction != target) & (target == int(label))
    harmed = changed & (base_prediction == target) & (target != int(label))
    transition = torch.zeros((4, 4), dtype=torch.long)
    if bool(changed.any().item()):
        encoded = 4 * base_prediction[changed].long() + int(label)
        transition = torch.bincount(encoded, minlength=16).reshape(4, 4).cpu()
    corrected_count = int(corrected.sum().item())
    harmed_count = int(harmed.sum().item())
    return {
        "selected_pixels": int(region_mask.sum().item()),
        "write_pixels": int(write_mask.sum().item()),
        "changed_pixels": int(changed.sum().item()),
        "corrected_pixels": corrected_count,
        "harmed_pixels": harmed_count,
        "net_gain_pixels": corrected_count - harmed_count,
        "raw_to_updated_transition": transition.tolist(),
    }
