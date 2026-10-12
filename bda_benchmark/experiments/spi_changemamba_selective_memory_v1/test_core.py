import torch

from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.core import (
    ChangeMambaTokenEncoder, balanced_event_folds, retrieval_relative_gap,
    threshold_candidates,
)


def test_key_never_requires_p3_state():
    dims = {
        "rgb_structure": 8, "change_local": 8, "change_semantic": 8,
        "multiring": 10, "geometry": 6, "p3_state": 20,
    }
    model = ChangeMambaTokenEncoder(dims, "k3_structural", key_dim=16, hidden_dim=32)
    groups = {name: torch.randn(4, dim) for name, dim in dims.items() if name != "p3_state"}
    key, logits = model(groups)
    assert key.shape == (4, 16)
    assert logits.shape == (4, 3)
    assert torch.allclose(key.norm(dim=1), torch.ones(4), atol=1e-5)


def test_relative_gap_prefers_predicted_class():
    result = {
        "indices": torch.tensor([[0, 1, 2]]),
        "scores": torch.tensor([[0.9, 0.8, 0.5]]),
        "prediction": torch.tensor([2]),
    }
    labels = torch.tensor([2, 2, 1])
    gap = retrieval_relative_gap(result, labels)
    assert torch.allclose(gap, torch.tensor([0.4]))


def test_balanced_event_folds_reduce_row_imbalance():
    events = ["large"] * 10 + ["medium"] * 6 + ["small"] * 2 + ["tiny"]
    folds = balanced_event_folds(events, 2)
    assert set().union(*folds) == set(events)
    assert folds[0].isdisjoint(folds[1])


def test_reject_all_threshold_exists():
    assert max(threshold_candidates()) > 1.0

