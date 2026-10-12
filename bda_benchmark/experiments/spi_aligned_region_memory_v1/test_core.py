import torch

from bda_benchmark.experiments.spi_aligned_region_memory_v1.core import (
    SoftRegionTokenizer, audit_contract, learned_region_loss,
)


def test_soft_region_shapes_and_gradients():
    model = SoftRegionTokenizer(embedding_dim=12, seed_rows=4, seed_cols=4)
    rgb = torch.randn(2, 3, 32, 32)
    assignment, feature = model(rgb, (16, 16))
    assert assignment.shape == (2, 16, 16, 16)
    assert torch.allclose(assignment.sum(1), torch.ones(2, 16, 16), atol=1e-5)
    loss, _ = learned_region_loss(assignment, feature, torch.randint(0, 2, (2, 32, 32)))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_contract_rejects_p3_shortcut_in_key():
    try:
        audit_contract({"key_groups": ["rgb", "p3_state"], "memory_write_allowed": True}, "train")
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("P3 shortcut was accepted")


def test_contract_rejects_validation_memory_write():
    try:
        audit_contract({"key_groups": ["rgb"], "memory_write_allowed": True}, "validation")
    except RuntimeError as error:
        assert "read-only" in str(error)
    else:
        raise AssertionError("validation Memory write was accepted")


def test_contract_rejects_majority_vote():
    try:
        audit_contract({"key_groups": ["rgb"], "memory_write_allowed": False, "majority_vote": True}, "validation")
    except RuntimeError as error:
        assert "majority" in str(error)
    else:
        raise AssertionError("hard majority vote was accepted")

