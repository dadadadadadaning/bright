from __future__ import annotations

import torch

from .model import SARFeatureAdapter


def test_all_modes_are_exact_identity_at_initialization():
    torch.manual_seed(3407)
    feature = torch.randn(2, 256, 24, 24)
    for mode in SARFeatureAdapter.MODES:
        adapter = SARFeatureAdapter(mode).eval()
        actual = adapter(feature)
        assert torch.equal(actual, feature), mode


def test_fsa_and_rcm_gates_receive_gradients():
    torch.manual_seed(3407)
    feature = torch.randn(2, 256, 12, 12)
    for mode, parameter_name in (("fsa", "fsa.output_gain"), ("rcm", "rcm.residual_gain")):
        adapter = SARFeatureAdapter(mode)
        adapter(feature).square().mean().backward()
        parameters = dict(adapter.named_parameters())
        gradient = parameters[parameter_name].grad
        assert gradient is not None
        assert torch.count_nonzero(gradient).item() > 0
