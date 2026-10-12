import torch

from bda_benchmark.model.PseudoInstanceHierarchicalMemory import (
    ProvenanceBalancedMemory, PseudoInstanceHierarchicalMemory,
    parent_mean_context, semantic_labels_from_delta,
)


def test_parent_context_is_broadcast_per_pseudo_instance():
    x = torch.tensor([[1., 0.], [3., 2.], [9., 4.]])
    out = parent_mean_context(x, torch.tensor([7, 7, 8]))
    assert torch.equal(out, torch.tensor([[2., 1.], [2., 1.], [9., 4.]]))


def test_semantic_label_is_recovered_from_delta_histogram():
    delta = torch.zeros(2, 4, 4, dtype=torch.long)
    # 1->2 cluster whose GT is mostly damaged; 2->3 whose GT is destroyed.
    delta[0, 2, 2] = 7; delta[0, 1, 2] = 2
    delta[1, 3, 3] = 9; delta[1, 2, 3] = 1
    labels = semantic_labels_from_delta(torch.tensor([6, 11]), delta)
    assert torch.equal(labels, torch.tensor([1, 2]))


def test_retrieval_excludes_same_event_and_image():
    memory = ProvenanceBalancedMemory(2, 2, 2, 4)
    memory.enqueue(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]),
                   torch.tensor([10, 20]), torch.tensor([100, 200]), torch.zeros(2, 2))
    found = memory.retrieve(torch.tensor([[1., 0.]]), torch.tensor([10]),
                            torch.tensor([999]), topk_per_class=1)
    assert not bool(found.valid[0, 0])
    assert bool(found.valid[0, 1])


def test_memory_is_class_balanced_by_capacity():
    memory = ProvenanceBalancedMemory(3, 2, 2, 2)
    keys = torch.randn(12, 2)
    labels = torch.tensor([0] * 8 + [1] * 2 + [2] * 2)
    memory.enqueue(keys, labels, torch.arange(12), torch.arange(12), torch.zeros(12, 2))
    assert torch.equal(memory.valid.sum(1), torch.tensor([2, 2, 2]))


def test_zero_retrieval_gate_starts_as_parametric_identity_and_gets_gradient():
    model = PseudoInstanceHierarchicalMemory(4, hidden=16, capacity_per_class=2, topk=1)
    x = torch.randn(3, 4)
    parent = torch.tensor([1, 1, 2]); direction = torch.tensor([6, 7, 11])
    area = torch.tensor([4., 5., 3.]); parent_area = torch.tensor([9., 9., 3.])
    event = torch.tensor([1, 1, 2]); image = torch.tensor([4, 4, 5])
    with torch.no_grad():
        embedding = model.encode(x, parent, direction)
        model.semantic_memory.enqueue(embedding, torch.tensor([0, 1, 2]),
                                      torch.tensor([8, 8, 8]), torch.tensor([9, 10, 11]),
                                      model.metadata(area, parent_area))
    no_memory = model(x, parent, direction, area, parent_area, event, image, False)
    with_memory = model(x, parent, direction, area, parent_area, event, image, True)
    assert torch.equal(no_memory["occurrence_logits"], with_memory["occurrence_logits"])
    with_memory["occurrence_logits"].sum().backward()
    assert model.retrieval_gate.grad is not None

