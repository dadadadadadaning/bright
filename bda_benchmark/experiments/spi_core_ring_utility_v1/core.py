from __future__ import annotations

from typing import Mapping

import torch

from bda_benchmark.experiments.spi_building_utility_v1.core import utility_counts


PROTOCOL = "SPI_CORE_RING_UTILITY_CACHE_v1"
ZONE_GEOMETRY_NAMES = ("is_core", "zone_fraction")
ZONE_SEMANTIC_NAMES = (
    "entropy_mean", "entropy_std", "entropy_q25", "entropy_q75",
    "margin_mean", "margin_std", "margin_q25", "margin_q75",
) + tuple(f"base_fraction_{index}" for index in range(4)) + tuple(
    f"candidate_fraction_{index}" for index in range(4)
)
ZONE_S2_NAMES = (
    "changed_fraction", "changed_l1_mean", "changed_l1_std", "changed_l1_q75",
) + tuple(f"direction_{before}_to_{after}" for before in range(4) for after in range(4))


def erode_four_connected(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary erosion with a 4-connected cross, implemented without new dependencies."""

    if mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    current = mask.bool().clone()
    for _ in range(radius):
        interior = current.clone()
        interior[0] = False
        interior[-1] = False
        interior[:, 0] = False
        interior[:, -1] = False
        interior[1:] &= current[:-1]
        interior[:-1] &= current[1:]
        interior[:, 1:] &= current[:, :-1]
        interior[:, :-1] &= current[:, 1:]
        current = interior
    return current


def split_core_ring(
    mask: torch.Tensor,
    radius: int = 1,
    minimum_core_area: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Partition one LocCC into an eroded core and its in-building boundary ring."""

    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("building mask must be non-empty")
    core = erode_four_connected(mask, radius)
    available = int(core.sum()) >= int(minimum_core_area)
    if not available:
        core = torch.zeros_like(mask)
    ring = mask & ~core
    if bool((core & ring).any()) or not torch.equal(core | ring, mask):
        raise RuntimeError("core/ring must be a disjoint partition of the LocCC")
    return core, ring, available


def deterministic_random_core(
    mask: torch.Tensor,
    core_area: int,
    seed: int,
) -> torch.Tensor:
    """Random in-building subset with exactly the same area as the true core."""

    result = torch.zeros_like(mask, dtype=torch.bool)
    locations = torch.nonzero(mask, as_tuple=False)
    if core_area <= 0:
        return result
    if core_area > locations.shape[0]:
        raise ValueError("core area exceeds building area")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chosen = locations[torch.randperm(locations.shape[0], generator=generator)[:core_area]]
    result[chosen[:, 0], chosen[:, 1]] = True
    return result


def _moments(values: torch.Tensor) -> torch.Tensor:
    values = values.float().flatten()
    if not values.numel():
        return torch.zeros(4)
    return torch.stack((
        values.mean(), values.std(unbiased=False),
        torch.quantile(values, 0.25), torch.quantile(values, 0.75),
    ))


def zone_feature_groups(
    *,
    zone_mask: torch.Tensor,
    building_mask: torch.Tensor,
    is_core: bool,
    base_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Prediction-only features for one core or ring; GT is deliberately absent."""

    if not bool(zone_mask.any()) or bool((zone_mask & ~building_mask).any()):
        raise ValueError("zone must be a non-empty subset of the building")
    zone_area = int(zone_mask.sum())
    building_area = int(building_mask.sum())
    probabilities = base_probabilities.float().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(0) / torch.log(torch.tensor(4.0))
    top2 = probabilities.topk(2, dim=0).values
    margin = top2[0] - top2[1]
    base_fraction = torch.bincount(base_prediction[zone_mask].long(), minlength=4).float() / zone_area
    candidate_fraction = (
        torch.bincount(candidate_prediction[zone_mask].long(), minlength=4).float() / zone_area
    )
    semantic = torch.cat((
        _moments(entropy[zone_mask]), _moments(margin[zone_mask]),
        base_fraction, candidate_fraction,
    ))

    changed = zone_mask & (base_prediction != candidate_prediction)
    changed_count = int(changed.sum())
    l1 = (candidate_probabilities.float() - base_probabilities.float()).abs().sum(0)
    changed_l1 = l1[changed]
    if changed_count:
        l1_stats = torch.stack((
            changed_l1.mean(), changed_l1.std(unbiased=False), torch.quantile(changed_l1, 0.75)
        ))
        transition = 4 * base_prediction[changed].long() + candidate_prediction[changed].long()
        transition_fraction = torch.bincount(transition, minlength=16).float() / changed_count
    else:
        l1_stats = torch.zeros(3)
        transition_fraction = torch.zeros(16)
    s2 = torch.cat((torch.tensor([changed_count / zone_area]), l1_stats, transition_fraction))
    return {
        "zone_geometry": torch.tensor([float(is_core), zone_area / building_area]),
        "zone_semantic": semantic.float(),
        "zone_s2": s2.float(),
    }


def apply_candidate_on_mask(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    output = base_prediction.clone()
    output[mask] = candidate_prediction[mask]
    return output


def update_zone_oracle(
    output: torch.Tensor,
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    zone_mask: torch.Tensor,
) -> Mapping[str, int | float]:
    counts = utility_counts(base_prediction, candidate_prediction, target, zone_mask)
    if int(counts["net"]) > 0:
        output[zone_mask] = candidate_prediction[zone_mask]
    return counts

