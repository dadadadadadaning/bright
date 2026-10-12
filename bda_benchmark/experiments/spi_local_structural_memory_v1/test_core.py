import torch

from .core import affected_target, binary_auroc, cross_event_topk, semantic_label_and_purity, weighted_binary_vote


def test_affected_target_is_hierarchical():
    assert torch.equal(affected_target(torch.tensor([0, 1, 2, 3])), torch.tensor([0, 0, 1, 1]))


def test_semantic_label_purity_is_evaluation_only_majority():
    label, purity = semantic_label_and_purity(torch.tensor([[1, 2], [2, 2]]), torch.tensor([[1, 1], [0, 1]], dtype=torch.bool))
    assert label == 2
    assert abs(purity - 2 / 3) < 1e-6


def test_cross_event_retrieval_excludes_query_event():
    query = torch.tensor([[1., 0.], [0., 1.]])
    memory = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    result = cross_event_topk(query, memory, ["a", "b"], ["a", "a", "b"], topk=1)
    # Query 0 (event a) may only use index 2 (event b). Query 1 (event b)
    # must exclude index 2, and index 1 is its exact feature match in event a.
    assert result.indices.tolist() == [[2], [1]]


def test_vote_and_auc_ordering():
    query = torch.tensor([[1., 0.], [0., 1.]])
    memory = torch.tensor([[1., 0.], [0., 1.]])
    result = cross_event_topk(query, memory, ["a", "b"], ["b", "a"], topk=1)
    vote = weighted_binary_vote(result, torch.tensor([0, 1]))
    assert torch.equal(vote, torch.tensor([0., 1.]))
    assert binary_auroc(torch.tensor([0, 1]), vote) == 1.
