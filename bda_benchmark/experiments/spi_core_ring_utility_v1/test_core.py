from __future__ import annotations

import torch

from .core import (
    ZONE_GEOMETRY_NAMES,
    ZONE_SEMANTIC_NAMES,
    ZONE_S2_NAMES,
    deterministic_random_core,
    split_core_ring,
    update_zone_oracle,
    zone_feature_groups,
)


def test_core_ring_is_exact_partition_and_random_control_matches_area():
    mask = torch.zeros((9, 9), dtype=torch.bool)
    mask[1:8, 1:8] = True
    core, ring, available = split_core_ring(mask, radius=1, minimum_core_area=4)
    assert available
    assert not bool((core & ring).any())
    assert torch.equal(core | ring, mask)
    random_core = deterministic_random_core(mask, int(core.sum()), seed=7341)
    assert int(random_core.sum()) == int(core.sum())
    assert not bool((random_core & ~mask).any())


def test_thin_region_has_no_core_and_full_ring():
    mask = torch.zeros((5, 7), dtype=torch.bool)
    mask[2, 1:6] = True
    core, ring, available = split_core_ring(mask, radius=1, minimum_core_area=1)
    assert not available
    assert not bool(core.any())
    assert torch.equal(ring, mask)


def test_zone_features_are_fixed_width_and_gt_free():
    building = torch.ones((4, 4), dtype=torch.bool)
    zone = torch.zeros_like(building)
    zone[1:3, 1:3] = True
    base = torch.ones((4, 4), dtype=torch.long)
    candidate = base.clone()
    candidate[zone] = 2
    base_prob = torch.nn.functional.one_hot(base, num_classes=4).permute(2, 0, 1).float()
    candidate_prob = torch.nn.functional.one_hot(candidate, num_classes=4).permute(2, 0, 1).float()
    groups = zone_feature_groups(
        zone_mask=zone, building_mask=building, is_core=True,
        base_probabilities=base_prob, candidate_probabilities=candidate_prob,
        base_prediction=base, candidate_prediction=candidate,
    )
    assert groups["zone_geometry"].numel() == len(ZONE_GEOMETRY_NAMES)
    assert groups["zone_semantic"].numel() == len(ZONE_SEMANTIC_NAMES)
    assert groups["zone_s2"].numel() == len(ZONE_S2_NAMES)
    assert all(torch.isfinite(value).all() for value in groups.values())


def test_zone_oracle_accepts_only_positive_net_zone():
    base = torch.ones((2, 4), dtype=torch.long)
    candidate = torch.full((2, 4), 2, dtype=torch.long)
    target = torch.tensor([[2, 2, 1, 1], [1, 1, 1, 1]])
    positive = torch.zeros_like(base, dtype=torch.bool)
    positive[0, :2] = True
    negative = ~positive
    output = base.clone()
    positive_counts = update_zone_oracle(output, base, candidate, target, positive)
    negative_counts = update_zone_oracle(output, base, candidate, target, negative)
    assert positive_counts["net"] > 0
    assert negative_counts["net"] < 0
    assert torch.equal(output[positive], candidate[positive])
    assert torch.equal(output[negative], base[negative])

