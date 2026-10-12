from __future__ import annotations

import torch

from .core import (
    cluster_delta_histogram,
    compress_channels,
    iter_transition_clusters,
    pool_role_tokens,
    role_masks,
    shuffled_cluster_mask,
    weighted_pool,
)


def test_clusters_respect_transition_and_connectivity() -> None:
    base = torch.zeros((5, 5), dtype=torch.long)
    candidate = base.clone()
    candidate[1, 1] = 1
    candidate[1, 2] = 1
    candidate[3, 3] = 2
    regions = torch.ones_like(base)
    clusters = list(iter_transition_clusters(base, candidate, regions))
    assert len(clusters) == 2
    assert sorted(int(item.cluster.sum()) for item in clusters) == [1, 2]
    assert sorted(item.transition for item in clusters) == [1, 2]


def test_role_masks_are_valid() -> None:
    building = torch.ones((9, 9), dtype=torch.bool)
    cluster = torch.zeros_like(building)
    cluster[4, 4] = True
    margin = torch.arange(81, dtype=torch.float32).reshape(9, 9)
    roles, metadata = role_masks(cluster, building, margin, ~cluster, halo_radius=1)
    assert not bool((roles["change"] & roles["anchor"]).any())
    assert not bool((roles["change"] & roles["boundary"]).any())
    assert int(roles["change"].sum()) == 1
    assert metadata["anchor_area"] > 0


def test_deterministic_channel_compression() -> None:
    feature = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(1, 8, 2, 2)
    compressed = compress_channels(feature, 4)
    assert compressed.shape == (1, 4, 2, 2)
    assert torch.allclose(compressed[:, 0], feature[:, :2].mean(1))


def test_weighted_pool_keeps_small_cluster() -> None:
    feature = torch.ones((1, 3, 2, 2))
    mask = torch.zeros((8, 8), dtype=torch.bool)
    mask[1, 1] = True
    pooled = weighted_pool(feature, mask)
    assert pooled.shape == (3,)
    assert torch.allclose(pooled, torch.ones(3))


def test_vectorized_role_pool_matches_individual_pooling() -> None:
    feature = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    masks = {}
    for index, name in enumerate(("change", "anchor", "boundary")):
        mask = torch.zeros((4, 4), dtype=torch.bool)
        mask[index, index:index + 2] = True
        masks[name] = mask
    expected = torch.stack([weighted_pool(feature, masks[name]) for name in masks])
    assert torch.allclose(pool_role_tokens(feature, masks), expected)


def test_histogram_delta_reconstructs_candidate() -> None:
    target = torch.tensor([[0, 1], [2, 3]])
    base = torch.tensor([[0, 0], [2, 1]])
    candidate = torch.tensor([[0, 1], [3, 3]])
    mask = base != candidate
    delta = cluster_delta_histogram(base, candidate, target, mask)
    base_hist = torch.bincount((target * 4 + base).reshape(-1), minlength=16).reshape(4, 4)
    candidate_hist = torch.bincount(
        (target * 4 + candidate).reshape(-1), minlength=16
    ).reshape(4, 4)
    assert torch.equal(base_hist + delta, candidate_hist)


def test_shuffled_control_preserves_area() -> None:
    building = torch.zeros((8, 8), dtype=torch.bool)
    building[1:7, 1:7] = True
    first = shuffled_cluster_mask(building, 7, 123)
    second = shuffled_cluster_mask(building, 7, 123)
    assert int(first.sum()) == 7
    assert torch.equal(first, second)
    assert not bool((first & ~building).any())
