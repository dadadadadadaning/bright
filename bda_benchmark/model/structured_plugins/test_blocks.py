import torch

from bda_benchmark.experiments.rgid_s9ic_clean.rgid.config import RGIDConfig
from bda_benchmark.model.structured_plugins.blocks import (
    AffectedOccurrenceResidual,
    CrossScaleBoundaryFusion,
    LocalCrossModalMatcher,
    RegionRelationBlock,
    StripSpatialBlock,
)
from bda_benchmark.model.structured_plugins.rgid_head import build_rgid_head


def test_strip_block_and_rgid_shapes():
    x = torch.randn(2, 16, 12, 15)
    assert StripSpatialBlock(16, strip_kernel=7)(x).shape == x.shape
    config = RGIDConfig(in_channels=16, hidden_channels=16, num_shared_blocks=1)
    output = build_rgid_head("strip", config, strip_kernel=7)(x)
    assert output["center_logits"].shape == (2, 1, 12, 15)
    assert output["boundary_logits"].shape == (2, 1, 12, 15)


def test_zero_gain_modules_are_exact_identities():
    high = torch.randn(2, 16, 8, 9)
    detail = torch.randn(2, 8, 16, 18)
    assert torch.equal(CrossScaleBoundaryFusion(16, 8)(high, detail), high)

    tokens = torch.randn(2, 7, 16)
    coordinates = torch.rand(2, 7, 2)
    assert torch.equal(RegionRelationBlock(16, 4)(tokens, coordinates), tokens)


def test_local_matcher_outputs_finite_maps():
    rgb = torch.randn(2, 8, 7, 9)
    sar = torch.randn(2, 8, 7, 9)
    output = LocalCrossModalMatcher(window_size=5)(rgb, sar)
    assert output.matched_sar.shape == sar.shape
    assert output.residual.shape == (2, 1, 7, 9)
    assert output.offset.shape == (2, 2, 7, 9)
    assert torch.isfinite(output.confidence).all()
    assert torch.isfinite(output.entropy).all()


def test_occurrence_residual_preserves_severity_and_support():
    logits = torch.randn(2, 4, 6, 7)
    evidence = torch.randn(2, 1, 6, 7)
    support = torch.zeros(2, 1, 6, 7)
    support[:, :, 2:5, 2:5] = 1
    module = AffectedOccurrenceResidual(maximum_shift=0.5)
    with torch.no_grad():
        module.gain.fill_(0.7)
    output = module(logits, evidence, support)
    assert torch.allclose(output[:, 3] - output[:, 2], logits[:, 3] - logits[:, 2])
    assert torch.equal(output[:, :2], logits[:, :2])
    outside = 1.0 - support
    assert torch.equal(output * outside, logits * outside)
