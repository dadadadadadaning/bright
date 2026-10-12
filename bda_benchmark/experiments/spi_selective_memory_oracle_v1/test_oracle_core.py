import torch

from .oracle_core import (
    OracleCandidate, force_occurrence_predictions, greedy_grouped_metric_oracle,
    greedy_metric_oracle, valid_miou,
)


def test_affected_proposal_preserves_severity_order():
    logits = torch.tensor([[4.0, 3.0, 1.0, 2.0], [0.0, 4.0, 3.0, 1.0]])
    prediction = force_occurrence_predictions(logits, True)
    assert prediction.tolist() == [3, 2]


def test_intact_proposal_does_not_change_nonaffected_pixels():
    logits = torch.tensor([[4.0, 3.0, 1.0, 2.0], [0.0, 1.0, 3.0, 2.0]])
    prediction = force_occurrence_predictions(logits, False)
    assert prediction.tolist() == [0, 1]


def test_greedy_oracle_is_monotonic():
    base = torch.tensor([
        [10, 0, 0, 0], [0, 8, 2, 0], [0, 3, 5, 0], [0, 0, 1, 5]
    ], dtype=torch.float64)
    good = torch.zeros(4, 4, dtype=torch.float64)
    good[2, 1], good[2, 2] = -1, 1
    bad = -good
    candidates = [
        OracleCandidate(0, "a", good, 1, 1, 0, 1, 0),
        OracleCandidate(1, "a", bad, 1, 0, 1, 0, 1),
    ]
    final, accepted, _ = greedy_metric_oracle(base, candidates)
    assert accepted == [0]
    assert valid_miou(final) > valid_miou(base)


def test_grouped_oracle_accepts_at_most_one_region_alternative():
    base = torch.tensor([
        [10, 0, 0, 0], [0, 8, 2, 0], [0, 3, 5, 0], [0, 0, 1, 5]
    ], dtype=torch.float64)
    first = torch.zeros(4, 4, dtype=torch.float64); first[2, 1], first[2, 2] = -1, 1
    second = torch.zeros(4, 4, dtype=torch.float64); second[3, 2], second[3, 3] = -1, 1
    rows = [
        OracleCandidate(7, "a", first, 1, 1, 0, 1, 0, source="memory"),
        OracleCandidate(7, "a", second, 1, 1, 0, 1, 0, source="direct"),
    ]
    _, accepted, _ = greedy_grouped_metric_oracle(base, rows)
    assert len(accepted) == 1
    assert accepted[0][0] == 7
