import torch

from bda_benchmark.model.margin_aware_affected import (
    ExactMarginAffectedResidual,
    MarginAwareAffectedEvidence,
    parent_decision_state,
)


def test_parent_state_reports_exact_required_margin():
    logits = torch.tensor([[[[2.0]], [[1.0]], [[0.5]], [[0.0]]]])
    state, required = parent_decision_state(logits)
    assert state.shape == (1, 10, 1, 1)
    assert torch.allclose(required, torch.tensor([[[[1.5]]]]))


def test_rejected_candidate_is_exact_parent():
    module = ExactMarginAffectedResidual(threshold=0.9)
    parent = torch.randn(1, 4, 2, 2)
    output, shift, accepted = module(parent, torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool), hard=True)
    assert torch.equal(output, parent)
    assert torch.count_nonzero(shift) == torch.count_nonzero(accepted) == 0


def test_accepted_candidate_crosses_occurrence_margin_and_preserves_severity():
    module = ExactMarginAffectedResidual(threshold=0.5, epsilon=0.01)
    parent = torch.tensor([[[[2.0]], [[1.0]], [[0.5]], [[0.0]]]])
    _, required = parent_decision_state(parent)
    output, shift, _ = module(parent, torch.tensor([[[[10.0]]]]), required, torch.ones(1, 1, 1, 1, dtype=torch.bool), hard=True)
    assert output[:, 2:4].amax(1).item() > output[:, 0:2].amax(1).item()
    assert torch.allclose(output[:, 3] - output[:, 2], parent[:, 3] - parent[:, 2])
    assert shift.item() > required.item()


def test_candidate_mask_excludes_parent_affected_pixels():
    module = MarginAwareAffectedEvidence(hidden=8)
    parent = torch.zeros(1, 4, 4, 4)
    parent[:, 2] = 2.0
    result = module(torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16), parent, torch.ones(1, 1, 4, 4))
    assert not result.candidate_mask.any()
