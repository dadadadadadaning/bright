import torch

from bda_benchmark.experiments.spi_building_relational_memory_v1.core import (
    VARIANTS, BuildingRelationalEncoder, audit_contract,
)


def fake_groups(batch=3, tokens=6):
    dimensions = {
        "rgb": 20, "sar": 18, "relation_unaligned": 22,
        "relation_aligned": 22, "multiring": 12, "geometry": 9,
        "sar_aligned": 12, "quality": 10, "p3_state": 16,
    }
    groups = {name: torch.randn(batch, tokens, dim) for name, dim in dimensions.items()}
    groups["geometry"][:, :, 4:6] = torch.rand(batch, tokens, 2)
    groups["quality"][:, :, 0] = torch.rand(batch, tokens)
    return groups, dimensions


def test_all_variants_forward_and_normalize():
    groups, dimensions = fake_groups()
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]).bool()
    for variant in VARIANTS:
        model = BuildingRelationalEncoder(dimensions, variant, node_dim=32, key_dim=24)
        key, logits, nodes = model(groups, mask)
        assert key.shape == (3, 24)
        assert logits.shape == (3, 3)
        assert nodes.shape == (3, 6, 32)
        assert torch.allclose(key.norm(dim=1), torch.ones(3), atol=1e-5)


def test_quality_is_not_a_semantic_branch():
    groups, dimensions = fake_groups()
    model = BuildingRelationalEncoder(dimensions, "m6_aligned_relation_quality", 32, 24)
    assert model.rgb is not None and model.sar is not None and model.relation is not None
    # There is deliberately no quality BranchEncoder: quality only gates relation attention.
    assert not hasattr(model, "quality")


def test_contract_rejects_shortcuts():
    try:
        audit_contract({"building_key_groups": ["rgb", "p3_state"]}, "train")
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("P3 shortcut was accepted")


def test_contract_rejects_validation_memory_write():
    try:
        audit_contract({"building_key_groups": ["rgb"], "memory_write_allowed": True}, "validation")
    except RuntimeError as error:
        assert "cannot write" in str(error)
    else:
        raise AssertionError("Validation Memory write was accepted")
