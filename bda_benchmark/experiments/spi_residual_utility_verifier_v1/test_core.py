import torch

from .core import apply_selected, feature_names, select_proposals, threshold_search, utility_weights


def row(key, source, affected, delta=1.0):
    matrix = torch.zeros(4, 4, dtype=torch.float64); matrix[2, 2] = delta
    return {"region_key": key, "source": source, "target_affected": affected, "delta": matrix}


def test_conflicting_low_margin_falls_back():
    rows = [row("x", "direct", True), row("x", "memory", False)]
    assert select_proposals(rows, [0.80, 0.75], threshold=0.5, conflict_margin=0.10) == []


def test_conflicting_high_margin_selects_best():
    rows = [row("x", "direct", True), row("x", "memory", False)]
    assert select_proposals(rows, [0.90, 0.60], threshold=0.5, conflict_margin=0.10) == [0]


def test_at_most_one_same_direction_proposal():
    rows = [row("x", "direct", True), row("x", "memory", True)]
    assert select_proposals(rows, [0.70, 0.80], threshold=0.5) == [1]


def test_apply_selected_adds_only_selected_delta():
    base = torch.eye(4, dtype=torch.float64)
    rows = [row("a", "direct", True, 2.0), row("b", "direct", True, 3.0)]
    assert torch.equal(apply_selected(base, rows, [1]), base + rows[1]["delta"])


def test_feature_contracts_are_nested():
    assert set(feature_names("m2_direct")) < set(feature_names("m3_class_memory"))


def test_utility_weights_are_bounded():
    value = utility_weights(torch.tensor([0.0, 1.0, 100.0]))
    assert float(value.min()) >= 0.10 and float(value.max()) <= 5.0

