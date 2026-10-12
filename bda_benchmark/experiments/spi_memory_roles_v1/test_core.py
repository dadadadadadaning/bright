import torch

from bda_benchmark.experiments.spi_memory_roles_v1.core import (
    MetricTeacher, RoleSupportAdapter, balanced_weights,
    cross_event_contrastive, hard_negative_margin, soft_cross_entropy,
)


def test_soft_cross_entropy_is_finite():
    logits = torch.randn(8, 4, requires_grad=True)
    target = torch.softmax(torch.randn(8, 4), 1)
    loss = soft_cross_entropy(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_cross_event_losses_have_gradient():
    raw = torch.randn(8, 16, requires_grad=True)
    embedding = torch.nn.functional.normalize(raw, dim=1)
    label = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    event = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = cross_event_contrastive(embedding, label, event) + hard_negative_margin(embedding, label, event)
    loss.backward()
    assert torch.isfinite(loss) and raw.grad is not None


def test_balanced_weights_sum_to_one():
    label = torch.tensor([0, 0, 0, 1, 2, 3])
    event = torch.tensor([1, 1, 2, 1, 2, 2])
    value = balanced_weights(label, event, torch.ones(6))
    assert torch.allclose(value.sum(), torch.tensor(1.0)) and (value > 0).all()


def test_role_adapter_is_exact_base_at_zero_gate_and_gate_gets_gradient():
    base_model = MetricTeacher(7, 8)
    adapter = RoleSupportAdapter(8, 6, base_model.classifier)
    base = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    query = torch.randn(3, 5, 6); support = torch.randn(3, 4, 5, 6)
    logits, refined = adapter(base, query, support)
    assert torch.equal(refined, base)
    assert torch.equal(logits, adapter.classifier(base))
    logits.sum().backward()
    assert adapter.gate.grad is not None

