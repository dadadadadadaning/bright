import torch

from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    class_fraction_maps, event_name, get_variant, reliable_token_targets,
)


def test_event_name():
    assert event_name("turkey-earthquake_123") == "turkey-earthquake"


def test_variant_controls_are_distinct():
    assert not get_variant("mm1_no_memory")["use_memory"]
    assert get_variant("mm2_memory")["use_memory"]
    assert get_variant("mm4_event_excluded")["exclude_event"]
    assert get_variant("mm5_shuffled_labels")["shuffle_labels"]


def test_reliable_targets_require_purity_and_building_prediction():
    damage = torch.ones(1, 4, 4, dtype=torch.long) * 2
    location = torch.ones_like(damage)
    p_building = torch.ones(1, 1, 2, 2)
    labels, reliable, purity = reliable_token_targets(damage, location, p_building)
    assert torch.all(labels == 2)
    assert torch.all(reliable)
    assert torch.allclose(purity, torch.ones_like(purity))


def test_class_fractions_do_not_treat_background_as_intact():
    damage = torch.tensor([[[0, 0], [1, 1]]])
    location = torch.tensor([[[0, 0], [1, 1]]])
    fractions, support = class_fraction_maps(damage, location, (1, 1))
    assert torch.isclose(fractions[0, 0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(support[0, 0, 0, 0], torch.tensor(0.5))
