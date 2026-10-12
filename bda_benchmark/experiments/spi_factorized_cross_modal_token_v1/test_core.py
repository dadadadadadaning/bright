from __future__ import annotations

import torch

from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    BuildingDeepSetEncoder,
    FactorizedKeyEncoder,
    local_alignment_diagnostics,
    recover_sar_display,
    retrieve_diverse_topk,
)
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.audit_complementarity import (
    correctness_partition,
)


GROUP_DIMS = {
    "rgb": 20,
    "sar": 21,
    "relation_unaligned": 22,
    "relation_aligned": 22,
    "multiring": 40,
    "geometry": 9,
    "quality": 10,
    "p3_state": 20,
}


def random_groups(batch: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {name: torch.randn(batch, dim, generator=generator) for name, dim in GROUP_DIMS.items()}


def test_zero_radius_alignment_is_identity() -> None:
    rgb = torch.randn(2, 8, 5, 7)
    sar = torch.randn_like(rgb)
    aligned, diagnostics = local_alignment_diagnostics(rgb, sar, radius=0)
    assert torch.equal(aligned, sar)
    assert torch.all(diagnostics["entropy"] == 0)
    assert torch.all(diagnostics["max_probability"] == 1)


def test_factorized_key_excludes_p3_quality_and_geometry() -> None:
    torch.manual_seed(11)
    model = FactorizedKeyEncoder(GROUP_DIMS, "t3_multiring").eval()
    left = random_groups()
    right = {name: value.clone() for name, value in left.items()}
    right["p3_state"] += 1000
    right["quality"] -= 1000
    right["geometry"] *= -1000
    with torch.inference_mode():
        left_key, left_logits = model(left)
        right_key, right_logits = model(right)
    assert torch.equal(left_key, right_key)
    assert torch.equal(left_logits, right_logits)


def test_concat_control_does_use_p3_state() -> None:
    torch.manual_seed(13)
    model = FactorizedKeyEncoder(GROUP_DIMS, "t0_concat").eval()
    left = random_groups()
    right = {name: value.clone() for name, value in left.items()}
    right["p3_state"] += 10
    with torch.inference_mode():
        left_key, _ = model(left)
        right_key, _ = model(right)
    assert not torch.allclose(left_key, right_key)


def test_diverse_retrieval_limits_each_event() -> None:
    query = torch.tensor([[1.0, 0.0]])
    memory = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.8, 0.2]])
    result = retrieve_diverse_topk(
        query,
        memory,
        torch.tensor([1, 1, 2, 2]),
        memory_events=["a", "a", "a", "b"],
        topk=2,
        max_per_event=1,
        exclude_same_image=False,
    )
    assert result["source_event_count"].item() == 2
    chosen_events = [["a", "a", "a", "b"][index] for index in result["indices"][0].tolist()]
    assert chosen_events == ["a", "b"]


def test_deepset_is_permutation_invariant() -> None:
    torch.manual_seed(17)
    model = BuildingDeepSetEncoder(token_dim=16, output_dim=8).eval()
    tokens = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    with torch.inference_mode():
        left = model(tokens, mask)
        right = model(tokens[:, permutation], mask[:, permutation])
    assert torch.allclose(left, right, atol=1e-6)


def test_recover_sar_display_uses_frozen_statistics() -> None:
    post = torch.zeros(1, 3, 2, 2)
    stats = {"normalization": {"p01": 0.0, "p99": 255.0, "mean": 123.675, "std": 10.0}}
    recovered = recover_sar_display(post, stats)
    assert torch.allclose(recovered, torch.zeros_like(recovered), atol=1e-5)


def test_three_way_correctness_partition_is_exhaustive() -> None:
    target = torch.ones(8, dtype=torch.long)
    # Enumerate every P3/Direct/Memory correctness combination exactly once.
    p3 = torch.tensor([2, 2, 2, 2, 1, 1, 1, 1])
    direct = torch.tensor([2, 2, 1, 1, 2, 2, 1, 1])
    memory = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1])
    result = correctness_partition(target, p3, direct, memory)
    assert sum(result["correctness_cube"].values()) == 8
    assert result["p3_wrong"]["total"] == 4
    assert result["p3_wrong"]["direct_only_correct"] == 1
    assert result["p3_wrong"]["memory_only_correct"] == 1
    assert result["p3_wrong"]["both_correct"] == 1
    assert result["p3_wrong"]["neither_correct"] == 1
    assert result["p3_wrong"]["oracle_union_recovery_rate"] == 0.75
    assert result["p3_wrong"]["memory_incremental_over_direct_rate"] == 0.5
