import torch

from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import (
    metrics_from_confusion,
    prediction_from_neighbours,
)
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    MetricKeyEncoder,
    cross_event_supervised_contrastive,
    local_tolerant_alignment,
    retrieve_global_topk,
    safe_region_to_pixel_writeback,
)


def test_local_alignment_is_identity_at_zero_radius():
    rgb = torch.randn(2, 8, 5, 7)
    sar = torch.randn_like(rgb)
    aligned, entropy, offset = local_tolerant_alignment(rgb, sar, radius=0)
    assert torch.equal(aligned, sar)
    assert not entropy.any() and not offset.any()


def test_metric_encoder_outputs_unit_keys():
    key, logits = MetricKeyEncoder(20, 8, 16)(torch.randn(7, 20))
    assert key.shape == (7, 8) and logits.shape == (7, 3)
    assert torch.allclose(key.norm(dim=1), torch.ones(7), atol=1e-5)


def test_retrieval_excludes_same_image():
    memory = torch.eye(3)
    result = retrieve_global_topk(
        memory[:1], memory, torch.tensor([1, 2, 3]), topk=1,
        query_images=["a"], memory_images=["a", "b", "c"],
    )
    assert int(result["indices"][0, 0]) != 0


def test_cross_event_loss_is_finite():
    key = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    label = torch.tensor([1, 1, 2, 2, 3, 3])
    loss = cross_event_supervised_contrastive(
        key, label, ["a", "b", "a", "b", "a", "b"],
        [str(i) for i in range(6)],
    )
    assert torch.isfinite(loss)


def test_writeback_preserves_background_and_zero_initialisation():
    logits = torch.randn(1, 4, 4, 4)
    ids = torch.zeros(1, 4, 4, dtype=torch.long)
    residual = torch.randn(1, 1, 3)
    confidence = torch.ones(1, 1)
    output = safe_region_to_pixel_writeback(
        logits, ids, residual, confidence, torch.tensor(0.0)
    )
    assert torch.equal(output, logits)
    output = safe_region_to_pixel_writeback(
        logits, ids, residual, confidence, torch.tensor(0.2)
    )
    assert torch.equal(output[:, 0], logits[:, 0])


def test_fixed_neighbours_can_be_relabelled_without_retrieval():
    indices = torch.tensor([[0, 1], [1, 2]])
    scores = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([1, 2, 3])
    prediction = prediction_from_neighbours(indices, scores, labels)
    assert prediction.tolist() == [1, 2]


def test_confusion_metrics_include_each_class_support():
    confusion = torch.tensor([[2, 0, 0], [0, 1, 1], [0, 0, 3]])
    metrics = metrics_from_confusion(confusion)
    assert metrics["support"] == [2, 2, 3]
    assert len(metrics["f1"]) == 3
