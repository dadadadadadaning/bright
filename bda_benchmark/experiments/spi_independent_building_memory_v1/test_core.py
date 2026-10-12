import torch

from .core import (ConditionalResidualModel, build_independent_card,
                   independent_evidence_maps, kcenter_indices, role_masks)
from .audit import RoleAdapter


def test_independent_maps_are_finite_and_do_not_need_p3():
    rgb = torch.randn(1, 3, 32, 32); sar = torch.randn(1, 2, 32, 32)
    r, s = independent_evidence_maps(rgb, sar, side=16)
    assert r.shape == (5, 16, 16) and s.shape == (6, 16, 16)
    assert torch.isfinite(r).all() and torch.isfinite(s).all()


def test_roles_partition_building_and_have_extremes():
    mask = torch.zeros(9, 9, dtype=torch.bool); mask[1:8, 1:8] = True
    anomaly = torch.arange(81).reshape(9, 9).float()
    roles = role_masks(mask, anomaly)
    assert torch.equal(roles["core"] | roles["ring"], mask)
    assert roles["anomaly"].any() and roles["stable"].any()


def test_card_shapes_and_finite():
    rgb = torch.randn(5, 16, 16); sar = torch.randn(6, 16, 16)
    mask = torch.zeros(32, 32, dtype=torch.bool); mask[4:28, 5:27] = True
    card = build_independent_card(rgb, sar, mask, torch.randn(8), tokens_per_role=3)
    assert card["role_tokens"].shape == (6, 3, 14)
    assert card["role_token_mask"].shape == (6, 3)
    assert all(torch.isfinite(v.float()).all() for v in card.values())


def test_zero_residual_is_exact_p3_probability():
    model = ConditionalResidualModel(7, hidden=8, prototypes_per_class=2)
    x = torch.randn(4, 7); p3 = torch.softmax(torch.randn(4, 4), 1)
    probability = model(x, p3)[0].softmax(1)
    assert torch.allclose(probability, p3, atol=1e-6)


def test_kcenter_is_deterministic_unique_and_bounded():
    x = torch.randn(20, 6, generator=torch.Generator().manual_seed(4))
    first = kcenter_indices(x, 7); second = kcenter_indices(x, 7)
    assert torch.equal(first, second) and len(first.unique()) == 7


def test_role_adapter_starts_at_p3_and_gate_gets_gradient():
    model = RoleAdapter(12, hidden=8)
    base = torch.softmax(torch.randn(3, 4), 1)
    query = torch.randn(3, 2, 6); support = torch.randn(3, 4, 2, 6)
    logits = model(base, query, support)
    assert torch.allclose(logits.softmax(1), base, atol=1e-6)
    logits.square().mean().backward()
    assert model.gate.grad is not None and float(model.gate.grad.abs()) > 0
