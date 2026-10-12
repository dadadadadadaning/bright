from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
ROLE_NAMES = ("change", "anchor", "boundary")
STAGE_NAMES = ("stage3", "stage4")


@dataclass(frozen=True)
class ClusterCrop:
    region_label: int
    cluster_index: int
    transition: int
    slices: tuple[slice, slice]
    building: torch.Tensor
    cluster: torch.Tensor

    @property
    def before(self) -> int:
        return self.transition // 4

    @property
    def after(self) -> int:
        return self.transition % 4


def iter_transition_clusters(
    base: torch.Tensor,
    candidate: torch.Tensor,
    region_labels: torch.Tensor,
) -> Iterator[ClusterCrop]:
    """Yield same-building, same-transition, 8-connected modification clusters."""

    if base.shape != candidate.shape or base.shape != region_labels.shape or base.ndim != 2:
        raise ValueError("base, candidate and region_labels must be aligned [H,W]")
    label_array = np.asarray(region_labels.cpu())
    for region_label, slices in enumerate(ndimage.find_objects(label_array), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        building = region_labels[y_slice, x_slice].cpu() == region_label
        base_crop = base[y_slice, x_slice].cpu()
        candidate_crop = candidate[y_slice, x_slice].cpu()
        changed = building & (base_crop != candidate_crop)
        if not bool(changed.any()):
            continue
        cluster_index = 0
        transitions = torch.unique(4 * base_crop[changed].long() + candidate_crop[changed].long())
        for transition in transitions.tolist():
            before, after = divmod(int(transition), 4)
            if before == after:
                continue
            selected = changed & (base_crop == before) & (candidate_crop == after)
            labels, count = ndimage.label(np.asarray(selected), structure=CONNECTIVITY_8)
            for component in range(1, int(count) + 1):
                mask = torch.from_numpy(labels == component)
                if bool(mask.any()):
                    yield ClusterCrop(
                        region_label=region_label,
                        cluster_index=cluster_index,
                        transition=int(transition),
                        slices=(y_slice, x_slice),
                        building=building,
                        cluster=mask,
                    )
                    cluster_index += 1


def role_masks(
    cluster: torch.Tensor,
    building: torch.Tensor,
    margin: torch.Tensor,
    unchanged: torch.Tensor,
    halo_radius: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Create cluster, stable in-building anchor and local boundary/halo roles."""

    if cluster.shape != building.shape or margin.shape != building.shape:
        raise ValueError("role inputs must share one cropped spatial shape")
    cluster_np = np.asarray(cluster.bool().cpu())
    building_np = np.asarray(building.bool().cpu())
    dilated_np = ndimage.binary_dilation(
        cluster_np, structure=CONNECTIVITY_8, iterations=max(int(halo_radius), 1)
    )
    boundary = torch.from_numpy(dilated_np & ~cluster_np)
    eligible = building.bool().cpu() & ~torch.from_numpy(dilated_np) & unchanged.bool().cpu()
    anchor = torch.zeros_like(building, dtype=torch.bool)
    if bool(eligible.any()):
        threshold = torch.median(margin.cpu()[eligible])
        anchor = eligible & (margin.cpu() >= threshold)
    if not bool(anchor.any()):
        anchor = building.bool().cpu() & ~cluster.bool().cpu()
    if not bool(anchor.any()):
        anchor = cluster.bool().cpu()
    inside_boundary = boundary & building.bool().cpu()
    metadata = {
        "anchor_fallback": float(not bool(eligible.any())),
        "anchor_area": float(anchor.sum()),
        "boundary_area": float(boundary.sum()),
        "boundary_inside_fraction": float(inside_boundary.sum()) / max(float(boundary.sum()), 1.0),
    }
    return {"change": cluster.bool().cpu(), "anchor": anchor, "boundary": boundary}, metadata


def shuffled_cluster_mask(building: torch.Tensor, area: int, seed: int) -> torch.Tensor:
    """Same-area random in-building mask used only as a spatial control."""

    indices = torch.nonzero(building.bool().reshape(-1), as_tuple=False).flatten()
    count = min(max(int(area), 0), int(indices.numel()))
    generator = torch.Generator().manual_seed(int(seed))
    selected = indices[torch.randperm(indices.numel(), generator=generator)[:count]]
    output = torch.zeros(building.numel(), dtype=torch.bool)
    output[selected] = True
    return output.reshape_as(building)


def embed_crop_mask(
    local_mask: torch.Tensor,
    slices: tuple[slice, slice],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    output = torch.zeros(output_shape, dtype=torch.bool)
    output[slices[0], slices[1]] = local_mask.bool().cpu()
    return output


def compress_channels(feature: torch.Tensor, output_channels: int = 32) -> torch.Tensor:
    """Deterministic channel grouping; no learned projection and no GT dependence."""

    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("feature must be [1,C,H,W]")
    channels = int(feature.shape[1])
    output_channels = min(int(output_channels), channels)
    if channels % output_channels == 0:
        return feature.float().reshape(
            1, output_channels, channels // output_channels, *feature.shape[-2:]
        ).mean(2)
    rows = feature.float().permute(0, 2, 3, 1).reshape(-1, 1, channels)
    rows = F.adaptive_avg_pool1d(rows, output_channels)
    return rows.reshape(1, *feature.shape[-2:], output_channels).permute(0, 3, 1, 2)


def to_nchw_stage(stage: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if stage.ndim != 4:
        raise ValueError("captured stage must be rank four")
    if stage.shape[1] == expected_channels:
        return stage.contiguous()
    if stage.shape[-1] == expected_channels:
        return stage.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        f"captured stage has shape {tuple(stage.shape)}, expected channel {expected_channels}"
    )


def weighted_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Area-resampled mask pooling that keeps very small clusters observable."""

    if feature.ndim != 4 or feature.shape[0] != 1 or mask.ndim != 2:
        raise ValueError("expected feature [1,C,h,w] and mask [H,W]")
    weight = F.interpolate(
        mask[None, None].float().to(feature.device),
        size=feature.shape[-2:],
        mode="area",
    )[0, 0].clamp_min(0.0)
    mass = weight.sum()
    if float(mass) <= 1e-8:
        weight = F.interpolate(
            mask[None, None].float().to(feature.device),
            size=feature.shape[-2:],
            mode="nearest",
        )[0, 0]
        mass = weight.sum().clamp_min(1.0)
    return (feature[0].float() * weight[None]).flatten(1).sum(1) / mass.clamp_min(1e-8)


def pool_map_channels(values: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return pool_role_tokens(values, masks).reshape(-1)


def pool_role_tokens(
    feature: torch.Tensor, masks: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Pool all three role masks in one tensor operation.

    The cache builder provides full-resolution feature maps, so this removes
    three repeated interpolation calls and several GPU synchronizations for
    every tiny modification cluster.
    """

    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("feature must be [1,C,H,W]")
    weights = torch.stack([masks[name].float() for name in ROLE_NAMES])[:, None]
    weights = weights.to(feature.device)
    if weights.shape[-2:] != feature.shape[-2:]:
        weights = F.interpolate(weights, size=feature.shape[-2:], mode="area")
    weights = weights[:, 0].clamp_min(0.0).flatten(1)
    mass = weights.sum(1, keepdim=True)
    pooled = weights @ feature[0].float().flatten(1).transpose(0, 1)
    return torch.where(mass > 1e-8, pooled / mass.clamp_min(1e-8), torch.zeros_like(pooled))


def pool_map_names(prefix: str, channel_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}_{role}_{channel}" for role in ROLE_NAMES for channel in channel_names)


def role_feature_vector(
    feature: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor]:
    pooled = pool_role_tokens(feature, masks)
    change, anchor, boundary = pooled.unbind(0)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a[None], b[None], dim=1, eps=1e-6)[0]

    summary = torch.stack(
        (
            change.mean(), change.std(unbiased=False), change.norm(),
            anchor.mean(), anchor.std(unbiased=False), anchor.norm(),
            boundary.mean(), boundary.std(unbiased=False), boundary.norm(),
            1.0 - cosine(change, anchor), (change - anchor).abs().mean(),
            1.0 - cosine(change, boundary), (change - boundary).abs().mean(),
        )
    )
    names = (
        "change_mean", "change_std", "change_norm",
        "anchor_mean", "anchor_std", "anchor_norm",
        "boundary_mean", "boundary_std", "boundary_norm",
        "change_anchor_cosine_distance", "change_anchor_l1",
        "change_boundary_cosine_distance", "change_boundary_l1",
    )
    memory = torch.cat((change - anchor, change - boundary))
    return summary, tuple(f"{prefix}_{name}" for name in names), memory


def spatial_probability_features(
    spec: ClusterCrop,
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    *,
    distance_map: torch.Tensor | None = None,
    margin_map: torch.Tensor | None = None,
    entropy_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    cluster = spec.cluster.bool()
    building = spec.building.bool()
    base = base_probabilities[:, spec.slices[0], spec.slices[1]].float()
    candidate = candidate_probabilities[:, spec.slices[0], spec.slices[1]].float()
    area = int(cluster.sum())
    building_area = int(building.sum())
    yy, xx = torch.nonzero(cluster, as_tuple=True)
    height, width = int(yy.max() - yy.min() + 1), int(xx.max() - xx.min() + 1)
    if distance_map is None:
        distance_map = torch.from_numpy(ndimage.distance_transform_edt(np.asarray(building)))
    distance = distance_map.float()
    selected_distance = distance[cluster]
    maximum_distance = max(float(distance.max()), 1.0)
    if margin_map is None:
        top2 = base.topk(2, dim=0).values
        margin = top2[0] - top2[1]
    else:
        margin = margin_map[spec.slices[0], spec.slices[1]].float()
    if entropy_map is None:
        entropy = -(base.clamp_min(1e-8) * base.clamp_min(1e-8).log()).sum(0) / math.log(4.0)
    else:
        entropy = entropy_map[spec.slices[0], spec.slices[1]].float()
    after = torch.full((area,), spec.after, dtype=torch.long)
    candidate_new = candidate[:, cluster].gather(0, after[None]).squeeze(0)
    base_new = base[:, cluster].gather(0, after[None]).squeeze(0)
    l1 = (candidate[:, cluster] - base[:, cluster]).abs().sum(0)
    direction = torch.zeros(16)
    direction[spec.transition] = 1.0
    values = torch.cat(
        (
            torch.tensor(
                (
                    math.log1p(area), area / max(building_area, 1),
                    area / max(height * width, 1), math.log((width + 1e-6) / (height + 1e-6)),
                    float(selected_distance.float().mean()) / maximum_distance,
                    float((selected_distance <= 2).float().mean()),
                    float(margin[cluster].mean()), float(margin[cluster].std(unbiased=False)),
                    float(entropy[cluster].mean()), float((candidate_new - base_new).mean()),
                    float(l1.mean()), float(l1.std(unbiased=False)),
                )
            ),
            direction,
        )
    ).float()
    names = (
        "log_area", "building_area_fraction", "bbox_fill", "log_aspect",
        "normalized_boundary_distance", "near_boundary_fraction",
        "base_margin_mean", "base_margin_std", "base_entropy_mean",
        "candidate_new_class_gain", "probability_l1_mean", "probability_l1_std",
        *(f"direction_{i // 4}_to_{i % 4}" for i in range(16)),
    )
    return values, tuple(names)


def utility_counts(
    base: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int | float]:
    valid = mask.bool() & (target >= 0) & (target <= 3)
    changed = valid & (base != candidate)
    corrected = changed & (candidate == target) & (base != target)
    harmed = changed & (base == target) & (candidate != target)
    neutral = changed & ~corrected & ~harmed
    count = int(changed.sum())
    net = int(corrected.sum()) - int(harmed.sum())
    return {
        "changed": count,
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "neutral": int(neutral.sum()),
        "net": net,
        "utility_ratio": net / max(count, 1),
    }


def cluster_delta_histogram(
    base: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool() & (target >= 0) & (target <= 3)
    truth = target[valid].long()
    base_index = truth * 4 + base[valid].long()
    candidate_index = truth * 4 + candidate[valid].long()
    base_hist = torch.bincount(base_index, minlength=16).reshape(4, 4)
    candidate_hist = torch.bincount(candidate_index, minlength=16).reshape(4, 4)
    return candidate_hist.long() - base_hist.long()


__all__ = [
    "ClusterCrop", "ROLE_NAMES", "STAGE_NAMES", "cluster_delta_histogram",
    "compress_channels", "embed_crop_mask", "iter_transition_clusters",
    "pool_map_channels", "pool_map_names", "pool_role_tokens", "role_feature_vector", "role_masks",
    "shuffled_cluster_mask", "spatial_probability_features", "to_nchw_stage",
    "utility_counts", "weighted_pool",
]
