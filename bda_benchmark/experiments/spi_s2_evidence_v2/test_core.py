from __future__ import annotations

import torch

from .core import direct_difference_evidence, partition_evidence, relation_evidence


def test_evidence_shapes_and_finiteness():
    torch.manual_seed(3407)
    rgb = torch.randn(1, 16, 12, 12)
    sar = torch.randn(1, 16, 12, 12)
    pre = torch.randn(1, 3, 48, 48)
    direct = direct_difference_evidence(rgb, sar, (48, 48))
    relation = relation_evidence(rgb, sar, (12, 12))
    grid = partition_evidence(pre, relation.values, (48, 48), mode="grid", cell_size=4)
    assert direct.values.shape == (1, 4, 48, 48)
    assert relation.values.shape == (1, 4, 12, 12)
    assert grid.values.shape == (1, 4, 48, 48)
    assert torch.isfinite(direct.values).all()
    assert torch.isfinite(relation.values).all()
    assert torch.isfinite(grid.values).all()


def test_identical_modalities_have_zero_relation_difference():
    torch.manual_seed(3407)
    feature = torch.randn(1, 16, 10, 10)
    relation = relation_evidence(feature, feature.clone(), (10, 10))
    assert torch.equal(relation.values, torch.zeros_like(relation.values))
