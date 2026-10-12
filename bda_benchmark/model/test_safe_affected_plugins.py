import torch

from bda_benchmark.model.safe_affected_plugins import (
    RGBInputStructureStem,
    SARInputTextureStem,
    SafeAffectedResidual,
    ShallowAffectedEvidence,
)


def test_shallow_stems_preserve_requested_resolution():
    rgb = RGBInputStructureStem(8)(torch.randn(2, 3, 32, 32), (8, 8))
    sar = SARInputTextureStem(8)(torch.randn(2, 3, 32, 32), (8, 8))
    assert rgb.shape == sar.shape == (2, 8, 8, 8)


def test_safe_residual_is_exact_parent_at_zero_gain():
    module = SafeAffectedResidual()
    parent = torch.randn(2, 4, 8, 8)
    output, delta = module(parent, torch.randn(2, 1, 8, 8), torch.ones(2, 1, 8, 8), torch.ones(2, 1, 8, 8))
    assert torch.equal(output, parent)
    assert torch.count_nonzero(delta) == 0


def test_safe_residual_preserves_severity_difference_when_open():
    module = SafeAffectedResidual()
    module.gain.data.fill_(0.3)
    parent = torch.randn(2, 4, 8, 8)
    output, delta = module(parent, torch.randn(2, 1, 8, 8), torch.ones(2, 1, 8, 8), torch.ones(2, 1, 8, 8))
    assert torch.all(delta >= 0)
    assert torch.allclose(output[:, 3] - output[:, 2], parent[:, 3] - parent[:, 2], atol=1e-6)


def test_projection_forbids_negative_gain():
    module = SafeAffectedResidual()
    module.gain.data.fill_(-1.0)
    module.project_parameters()
    assert module.gain.item() == 0.0


def test_zero_gain_can_open_in_the_safe_positive_direction():
    module = SafeAffectedResidual()
    parent = torch.zeros(1, 4, 4, 4)
    output, _ = module(
        parent,
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
    )
    (-output[:, 2:4].mean()).backward()
    assert module.gain.grad is not None
    assert module.gain.grad.item() < 0.0


def test_local_evidence_outputs_bounded_confidence():
    module = ShallowAffectedEvidence(hidden=8, local_match=True, confidence_gate=True)
    logits, gate, diagnostics = module(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32), (8, 8))
    assert logits.shape == gate.shape == (1, 1, 8, 8)
    assert gate.min() >= 0 and gate.max() <= 1
    assert diagnostics["match_offset"].shape == (1, 2, 8, 8)
