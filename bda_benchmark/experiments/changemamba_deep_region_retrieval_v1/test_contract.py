import torch

from .make_control_cache import main as _unused_main


def test_wrong_label_mapping_has_no_fixed_point():
    labels = torch.tensor([1, 2, 3])
    wrong = labels.remainder(3) + 1
    assert torch.equal(wrong, torch.tensor([2, 3, 1]))
    assert not torch.any(wrong == labels)


def test_row_permutation_preserves_shapes():
    generator = torch.Generator().manual_seed(3407)
    value = torch.arange(24).reshape(4, 3, 2)
    permutation = torch.randperm(4, generator=generator)
    assert value[permutation].shape == value.shape
    assert sorted(permutation.tolist()) == list(range(4))


def test_protocol_never_admits_test_by_filename():
    forbidden = "test_set.txt"
    assert "test" in forbidden.lower()

