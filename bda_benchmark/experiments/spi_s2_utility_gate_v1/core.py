from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
VARIANTS = (
    "u0_frozen_p3",
    "u1_fixed_s2",
    "u2_direction_policy",
    "u3_utility_gate",
    "o1_pixel_oracle",
)
GATE_INPUT_CHANNELS = 35


def batched_normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 4 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be [B,C,H,W] with C >= 2")
    probabilities = probabilities.float().clamp_min(1e-8)
    denominator = torch.log(probabilities.new_tensor(float(probabilities.shape[1])))
    return -(probabilities * probabilities.log()).sum(1, keepdim=True) / denominator


@dataclass
class S2Candidate:
    probabilities: torch.Tensor
    support: torch.Tensor
    region_features: torch.Tensor
    records: list[dict[str, float | int | bool]]


@dataclass(frozen=True)
class RegionLabelMap:
    """Dense LocCC labels: zero is background and 1..count are kept regions."""

    labels: torch.Tensor
    count: int


def predicted_building_label_map(
    localization_prediction: torch.Tensor,
    minimum_area: int = 32,
) -> RegionLabelMap:
    """Create one dense label map instead of one HxW mask per component."""

    if localization_prediction.ndim != 2:
        raise ValueError("localization_prediction must be [H,W]")
    if int(minimum_area) < 1:
        raise ValueError("minimum_area must be positive")
    from scipy import ndimage

    binary = localization_prediction.detach().cpu().numpy() > 0
    labels, component_count = ndimage.label(
        binary,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    areas = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)
    kept = np.flatnonzero(areas[1:] >= int(minimum_area)) + 1
    remap = np.zeros(int(component_count) + 1, dtype=np.int64)
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int64)
    dense = torch.from_numpy(remap[labels])
    return RegionLabelMap(labels=dense, count=int(len(kept)))


def _masks_to_label_map(
    regions: Sequence[torch.Tensor],
    spatial_shape: tuple[int, int],
) -> RegionLabelMap:
    """Compatibility adapter for tests and historical callers."""

    labels = torch.zeros(spatial_shape, dtype=torch.long)
    count = 0
    for raw_mask in regions:
        mask = raw_mask.detach().cpu().bool()
        if mask.shape != spatial_shape:
            raise ValueError("region mask and probabilities have different spatial shapes")
        if not bool(mask.any()):
            continue
        count += 1
        if bool((labels[mask] != 0).any()):
            raise ValueError("region masks must not overlap")
        labels[mask] = count
    return RegionLabelMap(labels=labels, count=count)


def fixed_s2_candidate_from_labels(
    base_probabilities: torch.Tensor,
    region_map: RegionLabelMap,
    config: SelectiveContextConfig | None = None,
) -> S2Candidate:
    """Vectorized, label-map equivalent of the locked S2 candidate."""

    config = config or SelectiveContextConfig()
    if base_probabilities.ndim != 3 or base_probabilities.shape[0] != 4:
        raise ValueError("base_probabilities must be [4,H,W]")
    probabilities = base_probabilities.float()
    labels = region_map.labels.to(probabilities.device, dtype=torch.long)
    if labels.shape != probabilities.shape[-2:]:
        raise ValueError("region label map and probabilities do not align")
    count = int(region_map.count)
    if count < 0 or (labels.numel() and int(labels.max()) > count):
        raise ValueError("invalid region label map")

    base_prediction = probabilities.argmax(0)
    output = probabilities.clone()
    support = torch.zeros_like(base_prediction, dtype=torch.bool)
    region_features = torch.zeros(
        (6, *base_prediction.shape), dtype=torch.float32, device=probabilities.device
    )
    if count == 0:
        return S2Candidate(output, support, region_features, [])

    flat_labels = labels.reshape(-1)
    flat_prediction = base_prediction.reshape(-1)
    areas = torch.bincount(flat_labels, minlength=count + 1)
    encoded = flat_labels * 4 + flat_prediction.long()
    votes = torch.bincount(encoded, minlength=(count + 1) * 4).reshape(count + 1, 4)

    probability_sums = torch.stack(
        [
            torch.bincount(
                flat_labels,
                weights=probabilities[class_id].reshape(-1),
                minlength=count + 1,
            )
            for class_id in range(4)
        ],
        dim=1,
    )
    means = probability_sums / areas.clamp_min(1).to(probability_sums.dtype)[:, None]
    foreground_votes = votes[:, 1:4]
    maximum_votes, majority_index = foreground_votes.max(dim=1)
    unique_majority = (foreground_votes == maximum_votes[:, None]).sum(dim=1) == 1
    unique_majority &= maximum_votes > 0
    mean_index = means[:, 1:4].argmax(dim=1)
    label_index = torch.where(unique_majority, majority_index, mean_index)
    chosen_votes = foreground_votes.gather(1, label_index[:, None]).squeeze(1)
    purity = chosen_votes.to(torch.float32) / areas.clamp_min(1).to(torch.float32)

    foreground_scores = means[:, 1:4]
    chosen_scores = foreground_scores.gather(1, label_index[:, None]).squeeze(1)
    competitors = foreground_scores.masked_fill(
        F.one_hot(label_index, num_classes=3).bool(), float("-inf")
    ).max(dim=1).values
    margin = chosen_scores - competitors
    accepted = (purity >= float(config.minimum_purity)) & (
        margin >= float(config.minimum_margin)
    )
    accepted[0] = False

    distributions = torch.zeros_like(means)
    foreground = foreground_scores / foreground_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
    distributions[:, 1:4] = foreground
    pixel_distribution = distributions[labels].permute(2, 0, 1)
    accepted_pixels = accepted[labels]
    alpha = float(config.blend_strength)
    mixed = (1.0 - alpha) * probabilities + alpha * pixel_distribution
    mixed /= mixed.sum(dim=0, keepdim=True).clamp_min(1e-8)
    output[:, accepted_pixels] = mixed[:, accepted_pixels]
    support = accepted_pixels

    image_area = float(base_prediction.numel())
    table = torch.stack(
        [
            purity,
            margin,
            torch.log1p(areas.to(torch.float32)) / np.log1p(image_area),
            distributions[:, 1],
            distributions[:, 2],
            distributions[:, 3],
        ],
        dim=1,
    )
    table[~accepted] = 0.0
    region_features = table[labels].permute(2, 0, 1).contiguous()

    changed_pixels = accepted_pixels & (output.argmax(0) != base_prediction)
    changed_counts = torch.bincount(
        labels[changed_pixels], minlength=count + 1
    )
    records = [
        {
            "region_index": index - 1,
            "accepted": bool(accepted[index]),
            "area": int(areas[index]),
            "purity": float(purity[index]),
            "margin": float(margin[index]),
            "changed_pixels": int(changed_counts[index]),
        }
        for index in range(1, count + 1)
    ]
    return S2Candidate(output, support, region_features, records)


def fixed_s2_candidate(
    base_probabilities: torch.Tensor,
    regions: Sequence[torch.Tensor],
    config: SelectiveContextConfig | None = None,
) -> S2Candidate:
    """Return the locked S2 soft candidate and prediction-only region metadata.

    This is intentionally label-free. RGB/SAR ground truth cannot enter S2.
    Region feature channels are purity, margin, log-area and three foreground
    class probabilities, broadcast only inside accepted LocCC regions.
    """

    region_map = _masks_to_label_map(regions, tuple(base_probabilities.shape[-2:]))
    return fixed_s2_candidate_from_labels(base_probabilities, region_map, config)


def direction_one_hot(base_prediction: torch.Tensor, candidate_prediction: torch.Tensor) -> torch.Tensor:
    if base_prediction.shape != candidate_prediction.shape:
        raise ValueError("prediction shapes do not match")
    encoded = 4 * base_prediction.long() + candidate_prediction.long()
    return F.one_hot(encoded, num_classes=16).permute(0, 3, 1, 2).float()


def utility_features(
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    support: torch.Tensor,
    region_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build G2 features without learned RGB/SAR, Memory or superpixels."""

    if base_probabilities.shape != candidate_probabilities.shape:
        raise ValueError("base and candidate probability shapes do not match")
    if base_probabilities.ndim != 4 or base_probabilities.shape[1] != 4:
        raise ValueError("probabilities must be [B,4,H,W]")
    if support.shape != base_probabilities[:, :1].shape:
        raise ValueError("support shape does not match probabilities")
    if region_features.shape != (base_probabilities.shape[0], 6, *base_probabilities.shape[-2:]):
        raise ValueError("region_features must be [B,6,H,W]")
    base = base_probabilities.float().clamp_min(1e-8)
    candidate = candidate_probabilities.float().clamp_min(1e-8)
    base_top2 = base.topk(2, dim=1).values
    base_confidence = base_top2[:, :1]
    base_margin = base_top2[:, :1] - base_top2[:, 1:2]
    candidate_confidence = candidate.max(1, keepdim=True).values
    base_prediction = base.argmax(1)
    candidate_prediction = candidate.argmax(1)
    changed = (base_prediction != candidate_prediction).unsqueeze(1) & support.bool()
    inputs = torch.cat(
        [
            base,
            candidate - base,
            batched_normalized_entropy(base),
            base_confidence,
            base_margin,
            (candidate - base).abs().sum(1, keepdim=True),
            candidate_confidence - base_confidence,
            direction_one_hot(base_prediction, candidate_prediction),
            region_features.float(),
        ],
        dim=1,
    )
    if inputs.shape[1] != GATE_INPUT_CHANNELS:
        raise AssertionError(f"expected {GATE_INPUT_CHANNELS} features, got {inputs.shape[1]}")
    return inputs, changed


class UtilityGate(nn.Module):
    """Pixelwise accept/reject head; zero init starts close to rejecting changes."""

    def __init__(self, inputs: int = GATE_INPUT_CHANNELS, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, inputs: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(inputs.float())) * changed.float()


def selector_targets(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (target >= 0) & (target <= 3)
    changed = valid & (base_prediction != candidate_prediction)
    positive = changed & (candidate_prediction == target) & (base_prediction != target)
    negative = changed & (base_prediction == target) & (candidate_prediction != target)
    decisive = positive | negative
    return positive[:, None].float(), decisive[:, None], positive, negative


def balanced_selector_bce(gate: torch.Tensor, target: torch.Tensor, decisive: torch.Tensor) -> torch.Tensor:
    positive = decisive & (target > 0.5)
    negative = decisive & ~positive
    positive_loss = -(torch.log(gate.clamp_min(1e-6)) * positive.float()).sum() / positive.sum().clamp_min(1)
    negative_loss = -(torch.log((1.0 - gate).clamp_min(1e-6)) * negative.float()).sum() / negative.sum().clamp_min(1)
    return 0.5 * (positive_loss + negative_loss)


def select_prediction(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    if accept.ndim == base_prediction.ndim + 1:
        accept = accept[:, 0]
    if not (base_prediction.shape == candidate_prediction.shape == accept.shape):
        raise ValueError("selection tensors do not align")
    return torch.where(accept.bool(), candidate_prediction, base_prediction)


def direction_policy_mask(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    policy: Mapping[str, bool],
) -> torch.Tensor:
    output = torch.zeros_like(base_prediction, dtype=torch.bool)
    for before in range(4):
        for after in range(4):
            if before != after and bool(policy.get(f"{before}->{after}", False)):
                output |= (base_prediction == before) & (candidate_prediction == after)
    return output


def empty_transition_audit() -> dict[str, object]:
    zeros = [[0 for _ in range(4)] for _ in range(4)]
    return {
        "changed": 0,
        "corrected": 0,
        "harmed": 0,
        "neutral": 0,
        "changed_matrix": [row[:] for row in zeros],
        "corrected_matrix": [row[:] for row in zeros],
        "harmed_matrix": [row[:] for row in zeros],
        "neutral_matrix": [row[:] for row in zeros],
        "soft_unchanged": {"count": 0, "improved": 0, "worsened": 0, "nll_delta_sum": 0.0},
    }


def compute_transition_audit(
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute one image once; the result can feed global and event totals."""

    audit = empty_transition_audit()
    base_prediction = base_probabilities.argmax(0)
    candidate_prediction = candidate_probabilities.argmax(0)
    valid = (target >= 0) & (target <= 3)
    changed = valid & (base_prediction != candidate_prediction)
    corrected = changed & (candidate_prediction == target) & (base_prediction != target)
    harmed = changed & (base_prediction == target) & (candidate_prediction != target)
    neutral = changed & ~corrected & ~harmed
    for name, mask in (("changed", changed), ("corrected", corrected), ("harmed", harmed), ("neutral", neutral)):
        count = int(mask.sum())
        audit[name] = int(audit[name]) + count
        if count:
            encoded = 4 * base_prediction[mask].long() + candidate_prediction[mask].long()
            matrix = torch.bincount(encoded, minlength=16).reshape(4, 4)
            old = torch.as_tensor(audit[f"{name}_matrix"], dtype=torch.long)
            audit[f"{name}_matrix"] = (old + matrix.cpu()).tolist()
    soft_mask = valid & (base_prediction == candidate_prediction)
    if support is not None:
        soft_mask &= support.bool()
    if bool(soft_mask.any()):
        safe_target = target.clamp(0, 3).long()
        base_true = base_probabilities.gather(0, safe_target[None]).squeeze(0).clamp_min(1e-8)
        candidate_true = candidate_probabilities.gather(0, safe_target[None]).squeeze(0).clamp_min(1e-8)
        delta = (-candidate_true.log() + base_true.log())[soft_mask]
        soft = audit["soft_unchanged"]
        soft["count"] += int(delta.numel())
        soft["improved"] += int((delta < -1e-7).sum())
        soft["worsened"] += int((delta > 1e-7).sum())
        soft["nll_delta_sum"] += float(delta.sum())
    return audit


def merge_transition_audit(
    destination: dict[str, object],
    source: Mapping[str, object],
) -> None:
    """Add a precomputed image audit without repeating pixel operations."""

    for name in ("changed", "corrected", "harmed", "neutral"):
        destination[name] = int(destination[name]) + int(source[name])
        destination_matrix = torch.as_tensor(destination[f"{name}_matrix"], dtype=torch.long)
        source_matrix = torch.as_tensor(source[f"{name}_matrix"], dtype=torch.long)
        destination[f"{name}_matrix"] = (destination_matrix + source_matrix).tolist()
    destination_soft = destination["soft_unchanged"]
    source_soft = source["soft_unchanged"]
    for name in ("count", "improved", "worsened"):
        destination_soft[name] = int(destination_soft[name]) + int(source_soft[name])
    destination_soft["nll_delta_sum"] = (
        float(destination_soft["nll_delta_sum"])
        + float(source_soft["nll_delta_sum"])
    )


def update_transition_audit(
    audit: dict[str, object],
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor | None = None,
) -> None:
    """Backward-compatible compute-and-merge entry point."""

    merge_transition_audit(
        audit,
        compute_transition_audit(
            base_probabilities,
            candidate_probabilities,
            target,
            support,
        ),
    )


def finalize_transition_audit(audit: Mapping[str, object]) -> dict[str, object]:
    output = dict(audit)
    changed = int(output["changed"])
    corrected = int(output["corrected"])
    harmed = int(output["harmed"])
    output["net"] = corrected - harmed
    output["harm_rate"] = harmed / max(changed, 1)
    output["correction_rate"] = corrected / max(changed, 1)
    output["decisive_precision"] = corrected / max(corrected + harmed, 1)
    rows = []
    changed_matrix = output["changed_matrix"]
    corrected_matrix = output["corrected_matrix"]
    harmed_matrix = output["harmed_matrix"]
    neutral_matrix = output["neutral_matrix"]
    for before in range(4):
        for after in range(4):
            if before == after:
                continue
            c = int(changed_matrix[before][after])
            good = int(corrected_matrix[before][after])
            bad = int(harmed_matrix[before][after])
            rows.append({
                "direction": f"{before}->{after}",
                "from": CLASS_NAMES[before],
                "to": CLASS_NAMES[after],
                "changed": c,
                "corrected": good,
                "harmed": bad,
                "neutral": int(neutral_matrix[before][after]),
                "net": good - bad,
                "decisive_precision": good / max(good + bad, 1),
            })
    output["directions"] = rows
    soft = dict(output["soft_unchanged"])
    soft["mean_nll_delta"] = soft["nll_delta_sum"] / max(soft["count"], 1)
    output["soft_unchanged"] = soft
    return output


def derive_direction_policy(
    global_audit: Mapping[str, object],
    event_audits: Mapping[str, Mapping[str, object]],
    *,
    minimum_pixels: int = 100,
    minimum_decisive_precision: float = 0.50,
    minimum_event_nonnegative_fraction: float = 0.60,
) -> dict[str, object]:
    global_rows = {row["direction"]: row for row in finalize_transition_audit(global_audit)["directions"]}
    event_rows = {
        event: {row["direction"]: row for row in finalize_transition_audit(audit)["directions"]}
        for event, audit in event_audits.items()
    }
    policy: dict[str, bool] = {}
    evidence: dict[str, object] = {}
    for direction, row in global_rows.items():
        observed = [table[direction] for table in event_rows.values() if table[direction]["changed"] > 0]
        nonnegative = sum(item["net"] >= 0 for item in observed)
        fraction = nonnegative / max(len(observed), 1)
        accept = bool(
            row["changed"] >= minimum_pixels
            and row["net"] > 0
            and row["decisive_precision"] >= minimum_decisive_precision
            and fraction >= minimum_event_nonnegative_fraction
        )
        policy[direction] = accept
        evidence[direction] = {
            **row,
            "events_observed": len(observed),
            "events_nonnegative": nonnegative,
            "event_nonnegative_fraction": fraction,
            "accepted": accept,
        }
    return {
        "policy": policy,
        "evidence": evidence,
        "thresholds": {
            "minimum_pixels": minimum_pixels,
            "minimum_decisive_precision": minimum_decisive_precision,
            "minimum_event_nonnegative_fraction": minimum_event_nonnegative_fraction,
        },
    }


def binary_ranking_metrics(scores: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, object]:
    if not scores:
        return {"auroc": None, "average_precision": None, "count": 0}
    score = np.concatenate(scores)
    label = np.concatenate(labels).astype(bool)
    positives = int(label.sum())
    negatives = int((~label).sum())
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None, "count": int(label.size), "positives": positives, "negatives": negatives}
    order = np.argsort(-score, kind="mergesort")
    ranked = label[order]
    tp = np.cumsum(ranked)
    fp = np.cumsum(~ranked)
    tpr = np.concatenate(([0.0], tp / positives))
    fpr = np.concatenate(([0.0], fp / negatives))
    precision = tp / np.arange(1, ranked.size + 1)
    return {
        "auroc": float(np.trapezoid(tpr, fpr)),
        "average_precision": float(precision[ranked].sum() / positives),
        "count": int(label.size),
        "positives": positives,
        "negatives": negatives,
    }
