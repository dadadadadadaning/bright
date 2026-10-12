import torch

from .core import (
    GroupDROState,
    event_losses,
    impact_weights,
    robust_loss,
    select_source_threshold,
)


def test_impact_weights_do_not_encode_event_frequency():
    utility = torch.tensor([0.01, 0.04, 0.16])
    weights = impact_weights(utility)
    assert torch.allclose(weights, torch.tensor([0.5, 1.0, 2.0]))


def test_event_balanced_loss_is_invariant_to_duplicate_easy_rows():
    logits = torch.tensor([0.0, 0.0, 0.0])
    labels = torch.tensor([1.0, 1.0, 0.0])
    weights = torch.ones(3)
    event_ids = torch.tensor([0, 0, 1])
    grouped = event_losses(logits, labels, weights, event_ids, 2)
    loss, effective = robust_loss(grouped, "event_balanced")
    assert torch.allclose(grouped[0], grouped[1])
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-6)
    assert torch.allclose(effective, torch.tensor([0.5, 0.5]))


def test_groupdro_increases_weight_for_harder_event():
    state = GroupDROState.create(2, decay=0.0, eta=1.0)
    loss, weights = robust_loss(
        torch.tensor([0.2, 1.0]), "groupdro", groupdro=state,
    )
    assert weights[1] > weights[0]
    assert loss > torch.tensor(0.6)


def test_threshold_selection_does_not_observe_heldout_event_scores():
    fixed = torch.zeros(3, 4, 4)
    for index in range(3):
        fixed[index] = torch.eye(4) * 10
    delta = torch.zeros(3, 4, 4)
    for index in range(3):
        delta[index, 1, 1] = 1
        delta[index, 1, 2] = -1
    cache = {
        "image_events": ["A", "B", "HELDOUT"],
        "image_names": ["a", "b", "c"],
        "image_index": torch.tensor([0, 1, 2]),
        "fixed_s2_histograms": fixed,
        "delta_hist": delta,
    }
    first = select_source_threshold(
        cache, torch.tensor([0.8, 0.7, 0.0]), ["A", "B"],
    )
    second = select_source_threshold(
        cache, torch.tensor([0.8, 0.7, 1.0]), ["A", "B"],
    )
    assert first["threshold"] == second["threshold"]
    assert first["pooled_gain_pp"] == second["pooled_gain_pp"]
