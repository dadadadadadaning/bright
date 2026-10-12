import torch

from bda_benchmark.model.model_centric_plugins import (
    HierarchicalOccurrenceSeverityDecoder,
    RGBStructureAdapter,
    RegistrationTolerantInteraction,
    SARScatteringFrequencyExperts,
)


def test_rgb_adapter_is_exact_identity_at_initialization():
    module = RGBStructureAdapter(16, hidden=8, strip_kernel=5)
    x = torch.randn(2, 16, 12, 12)
    assert torch.equal(module(x), x)


def test_sar_experts_are_exact_identity_and_routes_sum_to_one():
    module = SARScatteringFrequencyExperts(16, hidden=8)
    x = torch.randn(2, 16, 12, 12)
    output, diagnostics = module(x, return_diagnostics=True)
    assert torch.equal(output, x)
    assert torch.allclose(diagnostics["sar_expert_weights"].sum(1), torch.ones(2, 1, 1))


def test_local_interaction_is_exact_parent_at_initialization():
    module = RegistrationTolerantInteraction(16, 4, 8, window_size=3)
    rgb = torch.randn(2, 16, 8, 8)
    sar = torch.randn(2, 16, 8, 8)
    parent = torch.randn(2, 16, 8, 8)
    output = module(rgb, sar, parent)
    assert torch.equal(output.feature, parent)
    assert output.confidence.min() >= 0 and output.confidence.max() <= 1


def test_hierarchy_is_normalized_and_exact_parent_at_initialization():
    module = HierarchicalOccurrenceSeverityDecoder(16, 8)
    feature = torch.randn(2, 16, 8, 8)
    parent = torch.randn(2, 4, 8, 8)
    output = module(feature, parent)
    assert torch.equal(output["logits"], parent)
    probability = output["hierarchical_logits"].exp().sum(1)
    assert torch.allclose(probability, torch.ones_like(probability), atol=1e-5)


def test_all_zero_gains_receive_gradient():
    rgb = RGBStructureAdapter(16, hidden=8, strip_kernel=5)
    sar = SARScatteringFrequencyExperts(16, hidden=8)
    x = torch.randn(2, 16, 12, 12)
    loss = rgb(x).square().mean() + sar(x).square().mean()
    loss.backward()
    assert rgb.gain.grad is not None and torch.isfinite(rgb.gain.grad)
    assert sar.gain.grad is not None and torch.isfinite(sar.gain.grad)
