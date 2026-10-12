import torch

from .core import (
    DirectionConditionedOutcomeModel, expected_confusion_delta,
    neutral_truth_prior, outcome_counts, outcome_proportions,
)


def _cache():
    delta = torch.zeros(2, 4, 4, dtype=torch.int32)
    # 1->2: corrected=2, harmed=1, neutral truth=3 count=1.
    delta[0, 2, 2] = 2; delta[0, 2, 1] = -2
    delta[0, 1, 2] = 1; delta[0, 1, 1] = -1
    delta[0, 3, 2] = 1; delta[0, 3, 1] = -1
    # 2->1: corrected=1, harmed=2.
    delta[1, 1, 1] = 1; delta[1, 1, 2] = -1
    delta[1, 2, 1] = 2; delta[1, 2, 2] = -2
    return {
        "utility": {
            "changed": torch.tensor([4, 3]), "corrected": torch.tensor([2, 1]),
            "harmed": torch.tensor([1, 2]), "neutral": torch.tensor([1, 0]),
        },
        "direction_code": torch.tensor([6, 9]), "delta_hist": delta,
    }


def test_outcome_counts_partition_changed_pixels():
    cache = _cache()
    assert torch.equal(outcome_counts(cache).sum(1), torch.tensor([4., 3.]))
    proportions = outcome_proportions(cache)
    assert torch.allclose(
        proportions.sum(1), torch.ones(2, dtype=proportions.dtype)
    )


def test_direction_conditioning_changes_feature_slopes():
    model = DirectionConditionedOutcomeModel(2)
    with torch.no_grad():
        model.shared.weight.zero_(); model.shared.bias.zero_()
        model.direction_weight[6, 0, 0] = 2.0
    x = torch.tensor([[1., 0.], [1., 0.]])
    output = model(x, torch.tensor([6, 9]))
    assert output[0, 0] == 2 and output[1, 0] == 0


def test_expected_delta_reconstructs_known_cluster():
    cache = _cache()
    selected = torch.tensor([True, False])
    prior = neutral_truth_prior(cache, selected, smoothing=0.0)
    probabilities = outcome_proportions(cache)[:1]
    result = expected_confusion_delta(
        probabilities, cache["direction_code"][:1], torch.tensor([4]), prior
    )
    assert torch.allclose(result[0], cache["delta_hist"][0].double())


def test_expected_delta_preserves_histogram_mass():
    prior = torch.zeros(16, 4, dtype=torch.float64)
    prior[6, 0] = 1.0
    delta = expected_confusion_delta(
        torch.tensor([[.25, .25, .50]]), torch.tensor([6]), torch.tensor([20]), prior
    )
    assert torch.isclose(delta.sum(), torch.tensor(0., dtype=torch.float64))
    assert torch.allclose(delta.sum(2), torch.zeros(1, 4, dtype=torch.float64))
