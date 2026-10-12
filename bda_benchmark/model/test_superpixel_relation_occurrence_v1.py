import torch

from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.core import (
    candidate_token_targets, make_region_ids, pooled_region_feature, region_boundary,
    split_regions_by_building,
)
from bda_benchmark.model.safe_affected_plugins import SafeAffectedResidual


def test_slic_relation_ids_do_not_read_targets_and_keep_shape():
    image = torch.randn(2, 3, 24, 32)
    ids = make_region_ids(image, (12, 16), mode="grid", n_segments=24, compactness=8.)
    assert ids.shape == (2, 12, 16)
    assert ids.dtype == torch.long


def test_pooling_broadcasts_a_region_mean():
    feature = torch.tensor([[[[1., 3.], [5., 7.]]]])
    ids = torch.tensor([[[0, 0], [1, 1]]])
    value = pooled_region_feature(feature, ids)
    assert torch.equal(value, torch.tensor([[[[2., 2.], [6., 6.]]]]))
    assert region_boundary(ids).dtype == torch.bool


def test_candidate_target_is_one_per_token_and_rejects_mixed_tokens():
    parent = torch.zeros(1, 4, 2, 2); parent[:, 1] = 3.
    building = torch.ones(1, 1, 2, 2)
    damage = torch.tensor([[[2, 2, 1, 1], [2, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]]])
    location = torch.ones_like(damage)
    pos, neg = candidate_token_targets(parent, building, damage, location)
    assert pos.tolist() == [[[True, False], [False, True]]]
    assert neg.tolist() == [[[False, True], [True, False]]]


def test_partial_building_token_can_be_a_positive_candidate():
    parent = torch.zeros(1, 4, 1, 1); parent[:, 1] = 3.
    building = torch.ones(1, 1, 1, 1)
    damage = torch.tensor([[[2, 0], [0, 0]]])
    location = torch.tensor([[[1, 0], [0, 0]]])
    pos, neg = candidate_token_targets(parent, building, damage, location, min_building_fraction=.25)
    assert pos.item() is True
    assert neg.item() is False


def test_building_split_prevents_cross_boundary_pooling():
    feature = torch.tensor([[[[2., 2., 100., 100.]]]])
    ids = torch.zeros(1, 1, 4, dtype=torch.long)
    building = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.bool)
    split = split_regions_by_building(ids, building)
    pooled = pooled_region_feature(feature, split)
    assert torch.equal(pooled, feature)


def test_segment_count_scales_with_feature_area():
    small = make_region_ids(torch.randn(1, 3, 24, 24), (160, 160), mode="grid", n_segments=192, compactness=8.)
    large = make_region_ids(torch.randn(1, 3, 24, 24), (256, 256), mode="grid", n_segments=192, compactness=8.)
    small_mean = small.numel() / torch.unique(small).numel()
    large_mean = large.numel() / torch.unique(large).numel()
    assert abs(float(small_mean - large_mean)) / float(small_mean) < .25


def test_safe_residual_is_exact_parent_at_zero_gain_and_preserves_severity():
    parent = torch.randn(1, 4, 4, 4); module = SafeAffectedResidual()
    output, _ = module(parent, torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    assert torch.equal(output, parent)
    module.gain.data.fill_(.5)
    output, _ = module(parent, torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    # Zero gain above remains a strict bitwise identity.  With non-zero float32
    # additions, subtracting two independently rounded channels can differ by
    # one ULP although both received the exact same delta.
    torch.testing.assert_close(output[:, 3] - output[:, 2], parent[:, 3] - parent[:, 2], rtol=0., atol=1e-6)
