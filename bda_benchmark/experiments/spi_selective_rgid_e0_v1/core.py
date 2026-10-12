from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.method import SplitRiskEvidence
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    RegionLabelMap,
)
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig


VARIANTS = (
    "p3",
    "r0_loccc_s2",
    "r1_all_rgid_s2",
    "r2_selective_rgid_s2",
    "r3_random_matched_s2",
    "r4_split_oracle",
)


def threshold_evidence(
    evidence: Mapping[int, SplitRiskEvidence], threshold: float
) -> dict[int, SplitRiskEvidence]:
    """Apply a frozen scalar threshold without recomputing image evidence."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    output = {}
    for parent_id, row in evidence.items():
        structurally_valid = (
            row.child_count >= 2
            and row.peak_count >= 2
            and row.rejection_reason not in {
                "rgid_has_fewer_than_two_children",
                "insufficient_stable_centers",
            }
        )
        accepted = structurally_valid and row.risk_score >= threshold
        reason = "accepted" if accepted else (
            row.rejection_reason if not structurally_valid else "risk_below_threshold"
        )
        output[int(parent_id)] = replace(row, accepted=accepted, rejection_reason=reason)
    return output


def selected_parent_ids(
    evidence: Mapping[int, SplitRiskEvidence], threshold: float
) -> tuple[int, ...]:
    """Return accepted parent IDs without copying every evidence dataclass."""

    selected = threshold_evidence(evidence, threshold)
    return tuple(sorted(parent_id for parent_id, row in selected.items() if row.accepted))


def partition_from_selected_parents(
    parent_labels: np.ndarray | torch.Tensor,
    child_labels: np.ndarray | torch.Tensor,
    selected_parents: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of selective_partition.

    Region numeric IDs are allowed to differ; region membership and therefore
    fixed-S2 predictions are identical.
    """

    parents = np.asarray(torch.as_tensor(parent_labels).cpu(), dtype=np.int32)
    children = np.asarray(torch.as_tensor(child_labels).cpu(), dtype=np.int32)
    if parents.ndim != 2 or children.shape != parents.shape:
        raise ValueError("parent and child labels must be matching [H,W] arrays")
    support = parents > 0
    if not np.array_equal(support, children > 0):
        raise ValueError("parent and child partitions must conserve support")
    output = np.zeros_like(parents, dtype=np.int32)
    if not support.any():
        return output
    max_parent = int(parents.max())
    selected_lookup = np.zeros(max_parent + 1, dtype=bool)
    selected_array = np.asarray(selected_parents, dtype=np.int64)
    if selected_array.size:
        if selected_array.min() <= 0 or selected_array.max() > max_parent:
            raise ValueError("selected parent ID is outside the parent map")
        selected_lookup[selected_array] = True
    stride = int(children.max()) + 1
    encoded = parents.astype(np.int64) * stride
    split_pixels = support & selected_lookup[parents]
    encoded[split_pixels] += children[split_pixels]
    _, inverse = np.unique(encoded[support], return_inverse=True)
    output[support] = inverse.astype(np.int32) + 1
    return output


def random_selected_parent_ids(
    evidence: Mapping[int, SplitRiskEvidence], count: int, seed: int
) -> tuple[int, ...]:
    candidates = np.asarray(
        sorted(parent_id for parent_id, row in evidence.items() if row.child_count >= 2),
        dtype=np.int64,
    )
    if count < 0 or count > len(candidates):
        raise ValueError("random split count exceeds splittable parents")
    if count == 0:
        return ()
    generator = np.random.default_rng(int(seed))
    return tuple(sorted(int(value) for value in generator.choice(candidates, count, replace=False)))


def region_map(labels: np.ndarray | torch.Tensor) -> RegionLabelMap:
    dense = torch.as_tensor(labels).long().cpu()
    if dense.ndim != 2 or bool((dense < 0).any()):
        raise ValueError("labels must be a nonnegative [H,W] map")
    unique = torch.unique(dense)
    positive = unique[unique > 0]
    if len(positive) and not torch.equal(
        positive, torch.arange(1, len(positive) + 1, dtype=positive.dtype)
    ):
        remap = torch.zeros(int(dense.max()) + 1, dtype=torch.long)
        remap[positive] = torch.arange(1, len(positive) + 1)
        dense = remap[dense]
    return RegionLabelMap(dense, int(dense.max().item()) if dense.numel() else 0)


def s2_prediction(probabilities: torch.Tensor, labels) -> torch.Tensor:
    """Prediction-only equivalent of fixed S2.

    The historical helper also materialises a 6xHxW diagnostic feature map.
    Selective RGID calls S2 many times per image but only needs argmax labels,
    so constructing those diagnostics dominated runtime and peak RAM.
    """

    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("probabilities must be [4,H,W]")
    config = SelectiveContextConfig()
    probs = probabilities.detach().float().cpu()
    dense = torch.as_tensor(labels).long().cpu()
    if dense.shape != probs.shape[-2:] or bool((dense < 0).any()):
        raise ValueError("labels must be a nonnegative map aligned with probabilities")
    count = int(dense.max().item()) if dense.numel() else 0
    prediction = probs.argmax(0)
    if count == 0:
        return prediction

    flat_labels = dense.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    areas = torch.bincount(flat_labels, minlength=count + 1)
    votes = torch.bincount(
        flat_labels * 4 + flat_prediction,
        minlength=(count + 1) * 4,
    ).reshape(count + 1, 4)
    sums = torch.stack(
        [
            torch.bincount(
                flat_labels,
                weights=probs[class_id].reshape(-1),
                minlength=count + 1,
            )
            for class_id in range(4)
        ],
        dim=1,
    )
    means = sums / areas.clamp_min(1).float()[:, None]
    foreground_votes = votes[:, 1:4]
    maximum_votes, majority_index = foreground_votes.max(dim=1)
    unique_majority = (foreground_votes == maximum_votes[:, None]).sum(dim=1) == 1
    unique_majority &= maximum_votes > 0
    mean_index = means[:, 1:4].argmax(dim=1)
    label_index = torch.where(unique_majority, majority_index, mean_index)
    chosen_votes = foreground_votes.gather(1, label_index[:, None]).squeeze(1)
    purity = chosen_votes.float() / areas.clamp_min(1).float()

    foreground_scores = means[:, 1:4]
    chosen_scores = foreground_scores.gather(1, label_index[:, None]).squeeze(1)
    competitors = foreground_scores.masked_fill(
        torch.nn.functional.one_hot(label_index, num_classes=3).bool(),
        float("-inf"),
    ).max(dim=1).values
    accepted = (purity >= float(config.minimum_purity)) & (
        chosen_scores - competitors >= float(config.minimum_margin)
    )
    accepted[0] = False

    accepted_flat = accepted[flat_labels]
    selected_indices = torch.nonzero(accepted_flat, as_tuple=False).flatten()
    if selected_indices.numel() == 0:
        return prediction
    distributions = torch.zeros_like(means)
    foreground = foreground_scores / foreground_scores.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    distributions[:, 1:4] = foreground
    selected_distribution = distributions[flat_labels[selected_indices]].T
    mixed = (
        (1.0 - float(config.blend_strength)) * probs.reshape(4, -1)[:, selected_indices]
        + float(config.blend_strength) * selected_distribution
    )
    output = flat_prediction.clone()
    output[selected_indices] = mixed.argmax(dim=0)
    return output.view_as(prediction)


def split_oracle_prediction(
    loccc_prediction: torch.Tensor,
    split_prediction: torch.Tensor,
    target: torch.Tensor,
    parent_labels: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """GT-only upper bound: choose LocCC-S2 or RGID-S2 per parent."""
    parents = torch.as_tensor(parent_labels).long().cpu()
    loccc = loccc_prediction.long().cpu()
    split = split_prediction.long().cpu()
    truth = target.long().cpu()
    if not (parents.shape == loccc.shape == split.shape == truth.shape):
        raise ValueError("oracle inputs must have identical spatial shapes")
    output = loccc.clone()
    count = int(parents.max().item()) if parents.numel() else 0
    if count == 0:
        return output, {"rgid_selected": 0, "ties_kept_loccc": 0}
    valid = (parents > 0) & (truth != 255)
    parent_valid = parents[valid]
    valid_counts = torch.bincount(parent_valid, minlength=count + 1)
    loc_correct = torch.bincount(
        parent_valid,
        weights=(loccc[valid] == truth[valid]).float(),
        minlength=count + 1,
    )
    split_correct = torch.bincount(
        parent_valid,
        weights=(split[valid] == truth[valid]).float(),
        minlength=count + 1,
    )
    exists = valid_counts > 0
    choose_split = exists & (split_correct > loc_correct)
    ties = exists & (split_correct == loc_correct)
    choose_split[0] = False
    ties[0] = False
    output[choose_split[parents]] = split[choose_split[parents]]
    return output, {
        "rgid_selected": int(choose_split.sum()),
        "ties_kept_loccc": int(ties.sum()),
    }


def change_audit(candidate: torch.Tensor, reference: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    valid = target != 255
    changed = valid & (candidate != reference)
    corrected = changed & (candidate == target) & (reference != target)
    harmed = changed & (candidate != target) & (reference == target)
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "neutral": int((changed & ~corrected & ~harmed).sum()),
        "net": int(corrected.sum() - harmed.sum()),
    }


def event_from_name(name: str) -> str:
    return str(name).rsplit("_", 1)[0]
