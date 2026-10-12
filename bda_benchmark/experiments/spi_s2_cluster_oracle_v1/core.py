from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy import ndimage


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def utility_counts(
    base: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int]:
    valid = mask.bool() & (target >= 0) & (target <= 3)
    changed = valid & (base != candidate)
    corrected = changed & (candidate == target) & (base != target)
    harmed = changed & (base == target) & (candidate != target)
    neutral = changed & ~corrected & ~harmed
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "neutral": int(neutral.sum()),
        "net": int(corrected.sum()) - int(harmed.sum()),
    }


def _cluster_crops(
    base: torch.Tensor,
    candidate: torch.Tensor,
    building: torch.Tensor,
) -> list[tuple[int, torch.Tensor]]:
    """Return transition id and local component mask for one already-cropped building."""
    changed = building.bool() & (base != candidate)
    clusters: list[tuple[int, torch.Tensor]] = []
    for transition in torch.unique(4 * base[changed].long() + candidate[changed].long()).tolist():
        before, after = divmod(int(transition), 4)
        if before == after:
            continue
        selected = changed & (base == before) & (candidate == after)
        labels, count = ndimage.label(np.asarray(selected.cpu()), structure=CONNECTIVITY_8)
        for label in range(1, int(count) + 1):
            mask = torch.from_numpy(labels == label)
            if bool(mask.any()):
                clusters.append((int(transition), mask))
    return clusters


def transition_clusters(
    base: torch.Tensor,
    candidate: torch.Tensor,
    building: torch.Tensor,
) -> list[tuple[int, torch.Tensor]]:
    """Return full-size masks for testing/diagnostics while labeling only the building bbox."""

    if base.shape != candidate.shape or base.shape != building.shape or base.ndim != 2:
        raise ValueError("base, candidate and building must be aligned [H,W] tensors")
    coordinates = torch.nonzero(building, as_tuple=False)
    if not coordinates.numel():
        return []
    y0, x0 = coordinates.min(0).values.tolist()
    y1, x1 = (coordinates.max(0).values + 1).tolist()
    rows = []
    for transition, local in _cluster_crops(
        base[y0:y1, x0:x1], candidate[y0:y1, x0:x1], building[y0:y1, x0:x1]
    ):
        full = torch.zeros_like(building, dtype=torch.bool)
        full[y0:y1, x0:x1] = local
        rows.append((transition, full))
    return rows


def oracle_predictions(
    base: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    region_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build correct-count building, transition-cluster and restricted pixel oracles."""

    building_oracle = base.clone()
    cluster_oracle = base.clone()
    pixel_oracle = base.clone()
    valid = (target >= 0) & (target <= 3)
    corrected = valid & (base != candidate) & (candidate == target) & (base != target)
    pixel_oracle[corrected] = candidate[corrected]

    counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    label_array = np.asarray(region_labels.cpu())
    objects = ndimage.find_objects(label_array)
    for region_label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        local_labels = region_labels[y_slice, x_slice]
        building = local_labels == region_label
        base_crop = base[y_slice, x_slice]
        candidate_crop = candidate[y_slice, x_slice]
        target_crop = target[y_slice, x_slice]
        building_utility = utility_counts(base_crop, candidate_crop, target_crop, building)
        if building_utility["changed"] <= 0:
            continue
        counts["changed_buildings"] += 1
        if building_utility["net"] > 0:
            building_view = building_oracle[y_slice, x_slice]
            building_view[building] = candidate_crop[building]
            counts["building_oracle_accepted"] += 1
        elif building_utility["net"] == 0:
            counts["building_oracle_ties"] += 1

        clusters = _cluster_crops(base_crop, candidate_crop, building)
        counts["clusters_total"] += len(clusters)
        for transition, cluster in clusters:
            before, after = divmod(transition, 4)
            key = f"{before}_to_{after}"
            cluster_utility = utility_counts(base_crop, candidate_crop, target_crop, cluster)
            transition_counts[f"{key}_total"] += 1
            counts["cluster_changed_pixels"] += cluster_utility["changed"]
            counts["cluster_corrected_pixels"] += cluster_utility["corrected"]
            counts["cluster_harmed_pixels"] += cluster_utility["harmed"]
            counts["cluster_neutral_pixels"] += cluster_utility["neutral"]
            if cluster_utility["net"] > 0:
                cluster_view = cluster_oracle[y_slice, x_slice]
                cluster_view[cluster] = candidate_crop[cluster]
                counts["cluster_oracle_accepted"] += 1
                transition_counts[f"{key}_accepted"] += 1
            elif cluster_utility["net"] == 0:
                counts["cluster_oracle_ties"] += 1
                transition_counts[f"{key}_ties"] += 1
            else:
                counts["cluster_oracle_rejected"] += 1
                transition_counts[f"{key}_rejected"] += 1

    return building_oracle, cluster_oracle, pixel_oracle, {
        **dict(counts), "transition_counts": dict(transition_counts)
    }


__all__ = ["oracle_predictions", "transition_clusters", "utility_counts"]
