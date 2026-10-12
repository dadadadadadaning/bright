import torch

from .core import area_bin_masks, duplicate_fraction, hard_token_coverage


def test_area_bins_are_exclusive_and_complete():
    area = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 100])
    stacked = torch.stack(list(area_bin_masks(area).values()))
    assert torch.equal(stacked.sum(0), torch.ones_like(area))


def test_hard_token_coverage():
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[0, 0] = True
    mask[15, 15] = True
    assert hard_token_coverage(mask, (slice(0, 16), slice(0, 16)), (16, 16), (2, 2)) == 2


def test_duplicate_fraction_within_building():
    keys = torch.tensor([1, 1, 2, 2])
    values = torch.tensor([[1., 2.], [1., 2.], [1., 2.], [3., 4.]])
    assert duplicate_fraction(keys, values) == 0.25
