from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage


BASELINE_NAMES = (
    "log_area", "bbox_fill", "log_aspect", "changed_fraction",
    "base_entropy_changed", "base_margin_changed", "probability_l1_changed",
)
TOPOLOGY_NAMES = (
    "cluster_count_per_100px", "largest_cluster_fraction", "second_cluster_fraction",
    "cluster_size_entropy", "changed_compactness", "centroid_dispersion",
)
DISTANCE_NAMES = (
    "boundary_distance_mean", "boundary_distance_q25", "boundary_distance_q50",
    "boundary_distance_q75", "near_boundary_fraction", "deep_interior_fraction",
    "changed_centroid_offset",
)
DIRECTIONS = ((1, 2), (0, 2), (2, 1), (1, 3))
DIRECTION_NAMES = tuple(
    f"direction_{before}_to_{after}_{name}"
    for before, after in DIRECTIONS
    for name in ("fraction", "largest_cluster", "boundary_distance")
)
PROBABILITY_NAMES = (
    "base_margin_mean", "base_margin_std", "base_margin_q25",
    "candidate_gain_mean", "candidate_gain_q75", "probability_l1_mean",
    "probability_l1_q75", "changed_entropy_mean", "changed_vs_unchanged_margin",
)


def _components(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:].astype(np.float64)
    return labels, sizes


def topology_features(changed: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
    changed_np = np.asarray(changed.bool().cpu())
    building_np = np.asarray(building.bool().cpu())
    count = int(changed_np.sum())
    if count == 0:
        return torch.zeros(len(TOPOLOGY_NAMES))
    labels, sizes = _components(changed_np)
    ordered = np.sort(sizes)[::-1]
    probabilities = sizes / max(sizes.sum(), 1.0)
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    entropy /= max(math.log(max(len(sizes), 2)), 1e-6)
    eroded = ndimage.binary_erosion(changed_np, structure=np.ones((3, 3)))
    perimeter = int((changed_np & ~eroded).sum())
    compactness = float(4.0 * math.pi * count / max(perimeter * perimeter, 1))
    centers = ndimage.center_of_mass(changed_np, labels, range(1, len(sizes) + 1))
    yy, xx = np.nonzero(building_np)
    diagonal = math.hypot(yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
    if len(centers) <= 1:
        dispersion = 0.0
    else:
        centers = np.asarray(centers)
        weights = sizes / sizes.sum()
        mean = (centers * weights[:, None]).sum(0)
        dispersion = float((np.linalg.norm(centers - mean, axis=1) * weights).sum() / max(diagonal, 1.0))
    return torch.tensor((
        100.0 * len(sizes) / max(count, 1), ordered[0] / count,
        ordered[1] / count if len(ordered) > 1 else 0.0, entropy,
        min(compactness, 1.0), dispersion,
    ), dtype=torch.float32)


def distance_features(changed: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
    changed_np = np.asarray(changed.bool().cpu())
    building_np = np.asarray(building.bool().cpu())
    if not changed_np.any():
        return torch.zeros(len(DISTANCE_NAMES))
    distance = ndimage.distance_transform_edt(building_np)
    maximum = max(float(distance[building_np].max()), 1.0)
    selected = distance[changed_np] / maximum
    changed_center = np.asarray(ndimage.center_of_mass(changed_np))
    building_center = np.asarray(ndimage.center_of_mass(building_np))
    yy, xx = np.nonzero(building_np)
    diagonal = math.hypot(yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
    return torch.tensor((
        selected.mean(), np.quantile(selected, 0.25), np.quantile(selected, 0.50),
        np.quantile(selected, 0.75), float((distance[changed_np] <= 2).mean()),
        float((selected >= 0.50).mean()),
        float(np.linalg.norm(changed_center - building_center) / max(diagonal, 1.0)),
    ), dtype=torch.float32)


def direction_features(
    base_prediction: torch.Tensor, candidate_prediction: torch.Tensor,
    building: torch.Tensor,
) -> torch.Tensor:
    changed = building & (base_prediction != candidate_prediction)
    total = max(int(changed.sum()), 1)
    distance = ndimage.distance_transform_edt(np.asarray(building.cpu()))
    maximum_distance = max(float(distance.max()), 1.0)
    rows = []
    for before, after in DIRECTIONS:
        selected = changed & (base_prediction == before) & (candidate_prediction == after)
        count = int(selected.sum())
        if count == 0:
            rows.extend((0.0, 0.0, 0.0))
            continue
        _, sizes = _components(np.asarray(selected.cpu()))
        rows.extend((count / total, float(sizes.max() / count), float(distance[np.asarray(selected.cpu())].mean() / maximum_distance)))
    return torch.tensor(rows, dtype=torch.float32)


def probability_features(
    base_probabilities: torch.Tensor, candidate_probabilities: torch.Tensor,
    base_prediction: torch.Tensor, candidate_prediction: torch.Tensor,
    building: torch.Tensor,
) -> torch.Tensor:
    probabilities = base_probabilities.float().clamp_min(1e-8)
    changed = building & (base_prediction != candidate_prediction)
    unchanged = building & ~changed
    if not bool(changed.any()):
        return torch.zeros(len(PROBABILITY_NAMES))
    top2 = probabilities.topk(2, dim=0).values
    margin = top2[0] - top2[1]
    entropy = -(probabilities * probabilities.log()).sum(0) / math.log(4.0)
    rows = torch.nonzero(changed, as_tuple=True)
    new_class = candidate_prediction[changed].long()
    candidate_new = candidate_probabilities[:, changed].gather(0, new_class[None]).squeeze(0)
    base_new = base_probabilities[:, changed].gather(0, new_class[None]).squeeze(0)
    gain = candidate_new - base_new
    l1 = (candidate_probabilities[:, changed] - base_probabilities[:, changed]).abs().sum(0)
    unchanged_margin = margin[unchanged].mean() if bool(unchanged.any()) else margin[changed].mean()
    return torch.stack((
        margin[changed].mean(), margin[changed].std(unbiased=False), torch.quantile(margin[changed], 0.25),
        gain.mean(), torch.quantile(gain, 0.75), l1.mean(), torch.quantile(l1, 0.75),
        entropy[changed].mean(), margin[changed].mean() - unchanged_margin,
    )).float()


def baseline_features(
    building: torch.Tensor, changed: torch.Tensor, base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
) -> torch.Tensor:
    yy, xx = torch.nonzero(building, as_tuple=True)
    area = int(building.sum())
    height, width = int(yy.max() - yy.min() + 1), int(xx.max() - xx.min() + 1)
    probabilities = base_probabilities.float().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(0) / math.log(4.0)
    top2 = probabilities.topk(2, dim=0).values
    margin = top2[0] - top2[1]
    selected = changed if bool(changed.any()) else building
    l1 = (candidate_probabilities - base_probabilities).abs().sum(0)
    return torch.tensor((
        math.log1p(area) / math.log1p(building.numel()), area / max(height * width, 1),
        math.log((width + 1e-6) / (height + 1e-6)), int(changed.sum()) / max(area, 1),
        float(entropy[selected].mean()), float(margin[selected].mean()), float(l1[selected].mean()),
    ), dtype=torch.float32)


def shuffled_changed_mask(building: torch.Tensor, changed_count: int, seed: int) -> torch.Tensor:
    indices = torch.nonzero(building.reshape(-1), as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(int(seed))
    selected = indices[torch.randperm(len(indices), generator=generator)[:changed_count]]
    output = torch.zeros(building.numel(), dtype=torch.bool)
    output[selected] = True
    return output.reshape_as(building)


def feature_groups(
    *, building: torch.Tensor, base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor, base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor, shuffle_seed: int,
) -> dict[str, torch.Tensor]:
    building = building.bool().cpu()
    yy, xx = torch.nonzero(building, as_tuple=True)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    building = building[y0:y1, x0:x1]
    base_prediction = base_prediction.cpu()[y0:y1, x0:x1]
    candidate_prediction = candidate_prediction.cpu()[y0:y1, x0:x1]
    base_probabilities = base_probabilities.cpu()[:, y0:y1, x0:x1]
    candidate_probabilities = candidate_probabilities.cpu()[:, y0:y1, x0:x1]
    changed = building & (base_prediction != candidate_prediction)
    shuffled = shuffled_changed_mask(building, int(changed.sum()), shuffle_seed)
    return {
        "baseline": baseline_features(building, changed, base_probabilities, candidate_probabilities),
        "topology": topology_features(changed, building),
        "distance": distance_features(changed, building),
        "direction": direction_features(base_prediction.cpu(), candidate_prediction.cpu(), building),
        "probability": probability_features(base_probabilities, candidate_probabilities, base_prediction.cpu(), candidate_prediction.cpu(), building),
        "shuffled_topology": topology_features(shuffled, building),
        "shuffled_distance": distance_features(shuffled, building),
    }


FEATURE_NAMES = {
    "baseline": BASELINE_NAMES, "topology": TOPOLOGY_NAMES, "distance": DISTANCE_NAMES,
    "direction": DIRECTION_NAMES, "probability": PROBABILITY_NAMES,
    "shuffled_topology": TOPOLOGY_NAMES, "shuffled_distance": DISTANCE_NAMES,
}


def utility_counts(base: torch.Tensor, candidate: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask & (target >= 0) & (target <= 3)
    changed = valid & (base != candidate)
    corrected = changed & (candidate == target) & (base != target)
    harmed = changed & (base == target) & (candidate != target)
    neutral = changed & ~corrected & ~harmed
    count = int(changed.sum()); net = int(corrected.sum()) - int(harmed.sum())
    return {"changed": count, "corrected": int(corrected.sum()), "harmed": int(harmed.sum()),
            "neutral": int(neutral.sum()), "net": net, "utility_ratio": net / max(count, 1)}


__all__ = ["FEATURE_NAMES", "feature_groups", "utility_counts", "topology_features", "distance_features"]
