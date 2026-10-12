import torch

from bda_benchmark.experiments.spi_hierarchical_building_retrieval_v1.core import (
    aggregate_building_cards, binary_auroc, direction_name, event_diverse_topk,
)


def test_building_cards_keep_mean_and_spread():
    feature = torch.tensor([[1., 0.], [3., 2.], [10., 4.]])
    parent = torch.tensor([7, 7, 9])
    card, unique, inverse = aggregate_building_cards(
        feature, parent, torch.tensor([1., 1., 1.])
    )
    assert unique.tolist() == [7, 9]
    assert inverse.tolist() == [0, 0, 1]
    assert torch.allclose(card[0, :2], torch.tensor([2., 1.]))
    assert torch.allclose(card[0, 2:4], torch.tensor([1., 1.]))


def test_event_diverse_topk_caps_each_event():
    indices = torch.tensor([0, 1, 2, 3, 4])
    scores = torch.tensor([.9, .8, .7, .6, .5])
    events = torch.tensor([1, 1, 1, 2, 3])
    selected, _ = event_diverse_topk(
        indices, scores, events, topk=3, maximum_per_event=1
    )
    assert selected.tolist() == [0, 3, 4]


def test_binary_auroc_is_one_for_perfect_order():
    assert binary_auroc(torch.tensor([0, 1, 0, 1]), torch.tensor([.1, .8, .2, .9])) == 1.0


def test_direction_name_uses_p3_to_s2_order():
    assert direction_name(6) == "intact_to_damaged"
    assert direction_name(11) == "damaged_to_destroyed"
