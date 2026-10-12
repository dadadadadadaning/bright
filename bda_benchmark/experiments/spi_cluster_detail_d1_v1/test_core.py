import torch

from .core import (
    fixed_edge_maps, fixed_sar_texture_maps, selective_histogram_from_fixed,
    structural_relation_features,
)


def test_fixed_maps_are_finite_and_parameter_free():
    value = torch.randn(1, 1, 16, 16)
    edge, edge_names = fixed_edge_maps(value)
    texture, texture_names = fixed_sar_texture_maps(value)
    assert edge.shape == (1, 5, 16, 16)
    assert texture.shape == (1, 6, 16, 16)
    assert len(edge_names) == 5 and len(texture_names) == 6
    assert torch.isfinite(edge).all() and torch.isfinite(texture).all()


def test_structure_relation_is_zero_for_identical_modalities():
    maps = torch.randn(1, 5, 8, 8)
    masks = {
        "change": torch.zeros(8, 8, dtype=torch.bool),
        "anchor": torch.zeros(8, 8, dtype=torch.bool),
        "boundary": torch.zeros(8, 8, dtype=torch.bool),
    }
    masks["change"][1:3, 1:3] = True
    masks["anchor"][5:7, 5:7] = True
    masks["boundary"][3:5, 3:5] = True
    values, names = structural_relation_features(maps, maps, masks)
    assert len(values) == len(names) == 24
    # Every third four-channel block is the cross-modal relation gap.
    assert torch.allclose(values[:4], torch.zeros(4), atol=1e-6)
    assert torch.allclose(values[12:16], torch.zeros(4), atol=1e-6)


def test_rejected_cluster_reverts_fixed_s2_delta():
    fixed = torch.zeros(2, 4, 4)
    fixed[:, 0, 0] = 10
    delta = torch.zeros(2, 4, 4)
    delta[0, 0, 0] = 2
    delta[1, 0, 0] = 3
    result = selective_histogram_from_fixed(fixed, delta, torch.tensor([True, False]))
    assert result[0, 0] == 17


def test_accept_all_is_exact_fixed_s2():
    fixed = torch.randn(3, 4, 4)
    delta = torch.randn(5, 4, 4)
    result = selective_histogram_from_fixed(fixed, delta, torch.ones(5, dtype=torch.bool))
    assert torch.allclose(result, fixed.sum(0).double())
