import numpy as np
import torch

from bda_benchmark.experiments.spi_cut_band_evidence_v1.core import (
    candidate_cut_band,
    compressed_feature_channels,
    scharr_magnitude,
    split_utility,
)


def test_cut_band_excludes_outer_outline_and_detects_internal_cut():
    parent = np.zeros((7, 7), dtype=np.int32)
    parent[1:6, 1:6] = 1
    child = parent.copy()
    child[1:6, 4:6] = 2
    band = candidate_cut_band(parent, child, 1, radius=0)
    assert band[:, 3:5].any()
    assert not band[1, 1]
    assert not band[5, 5]


def test_cut_band_dilation_conserves_parent_support():
    parent = np.ones((9, 9), dtype=np.int32)
    child = np.ones((9, 9), dtype=np.int32)
    child[:, 5:] = 2
    band = candidate_cut_band(parent, child, 1, radius=2)
    assert band.any()
    assert torch.all(band <= torch.from_numpy(parent == 1))


def test_scharr_is_parameter_free_and_edge_sensitive():
    image = torch.zeros(1, 3, 16, 16)
    image[:, :, :, 8:] = 1
    edge = scharr_magnitude(image)
    assert edge.shape == (1, 1, 16, 16)
    assert torch.isfinite(edge).all()
    assert edge[:, :, :, 7:9].mean() > edge[:, :, :, :4].mean()


def test_split_utility_counts_only_changed_pixels():
    loccc = torch.tensor([[1, 1], [1, 1]])
    split = torch.tensor([[1, 2], [1, 2]])
    target = torch.tensor([[1, 2], [2, 1]])
    row = split_utility(loccc, split, target, torch.ones_like(target).bool())
    assert row["changed"] == 2
    assert row["corrected"] == 1
    assert row["harmed"] == 1
    assert row["net"] == 0


def test_channel_compression_is_deterministic_and_reduces_shape():
    feature = torch.arange(2 * 8 * 3 * 3).reshape(2, 8, 3, 3).float()
    first = compressed_feature_channels(feature, 4)
    second = compressed_feature_channels(feature, 4)
    assert first.shape == (2, 4, 3, 3)
    assert torch.equal(first, second)
