# -*- coding: utf-8 -*-
import unittest

import torch

from model.DamageFormer_S9_XHPI import (
    CrossModalMetricProjector,
    HierarchicalPrototypeInstanceMemory,
    MetricConfig,
    RetrievalConfig,
    RetrievalResult,
    build_hierarchical_memory,
    hierarchical_log_probs,
    metric_training_loss,
    safe_probability_fusion,
)


class S9XHPITests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_hierarchical_probabilities_normalize(self):
        exist = torch.randn(9, 2)
        severity = torch.randn(9, 2)
        probability = hierarchical_log_probs(exist, severity).exp()
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones(9), atol=1e-6))

    def test_shared_private_metric_and_local_receive_gradients(self):
        model = CrossModalMetricProjector(MetricConfig(raw_dim=8, embed_dim=8, hidden_dim=16))
        batch = {
            "rgb_structure": torch.randn(12, 8),
            "sar_global": torch.randn(12, 8),
            "sar_ring": torch.randn(12, 8),
            "change_global": torch.randn(12, 8),
            "local_summary": torch.randn(12, 8),
        }
        labels = torch.tensor([0, 1, 2] * 4)
        events = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        output = model(batch)
        loss, values = metric_training_loss(output, labels, events)
        loss.backward()
        self.assertGreater(values["total"], 0)
        self.assertIsNotNone(model.local_proj[-1].weight.grad)
        self.assertGreater(float(model.change_proj[-1].weight.grad.abs().sum()), 0)

    def test_train_only_balanced_memory_has_prototypes_and_instances(self):
        dim = 8
        count = 36
        labels = torch.tensor([0, 1, 2] * 12)
        events = [f"event_{index % 3}" for index in range(count)]
        local = torch.randn(count * 2, dim)
        payload = {
            "rgb_structure": torch.randn(count, dim),
            "sar_global": torch.randn(count, dim),
            "sar_ring": torch.randn(count, dim),
            "change_global": torch.randn(count, dim) + torch.nn.functional.one_hot(labels, 3).float() @ torch.randn(3, dim),
            "local_tokens": local,
            "local_offsets": torch.arange(0, count * 2 + 1, 2),
            "base_probs": torch.softmax(torch.randn(count, 3), 1),
            "labels": labels,
            "events": events,
            "sample_ids": [f"train_{i}" for i in range(count)],
            "metadata": {"split": "train", "source_split": "train_only"},
        }
        projector = CrossModalMetricProjector(MetricConfig(dim, dim, 16))
        memory = build_hierarchical_memory(
            payload,
            projector,
            prototypes_per_class=2,
            max_per_event_class=3,
            hard_fraction=0.2,
        )
        self.assertEqual(memory["prototypes"].shape[0], 6)
        self.assertEqual(memory["local_offsets"].numel(), memory["labels"].numel() + 1)
        pair_counts = {}
        for label, event in zip(memory["labels"].tolist(), memory["events"]):
            pair_counts[(label, event)] = pair_counts.get((label, event), 0) + 1
        self.assertLessEqual(max(pair_counts.values()), 3)

    def test_prototype_routes_then_instances_vote(self):
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        payload = {
            "structure": prototypes.repeat_interleave(2, 0),
            "sar": prototypes.repeat_interleave(2, 0),
            "change": prototypes.repeat_interleave(2, 0),
            "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
            "events": ["a", "b", "a", "b", "a", "b"],
            "local_tokens": prototypes.repeat_interleave(2, 0),
            "local_offsets": torch.arange(7),
            "prototypes": prototypes,
            "prototype_labels": torch.tensor([0, 1, 2]),
            "prototype_assignment": torch.tensor([0, 0, 1, 1, 2, 2]),
        }
        memory = HierarchicalPrototypeInstanceMemory(payload, torch.device("cpu"))
        query = {
            "structure": torch.tensor([0.0, 1.0]),
            "sar": torch.tensor([0.0, 1.0]),
            "change": torch.tensor([0.0, 1.0]),
            "local": torch.tensor([[0.0, 1.0]]),
        }
        config = RetrievalConfig(
            route_top_prototypes=1,
            topk=2,
            rerank_candidates=2,
            similarity_threshold=0.5,
            purity_threshold=0.5,
            retrieval_margin_threshold=0.0,
            agreement_threshold=0.2,
            base_margin_max=1.0,
            minimum_events=2,
        )
        result = memory.search(query, torch.tensor([0.34, 0.33, 0.33]), config)
        self.assertEqual(int(result.probabilities.argmax()), 1)
        self.assertTrue(result.active)

    def test_bypass_is_exact_and_active_fusion_is_convex(self):
        base = torch.tensor([0.8, 0.1, 0.1])
        retrieval = RetrievalResult(
            probabilities=torch.tensor([0.0, 1.0, 0.0]),
            prototype_probabilities=torch.tensor([0.0, 1.0, 0.0]),
            neighbor_indices=torch.tensor([0]),
            neighbor_scores=torch.tensor([1.0]),
            gate=0.5,
            active=True,
            purity=1.0,
            margin=1.0,
            best_similarity=1.0,
            agreement=1.0,
            event_count=2,
        )
        self.assertTrue(torch.allclose(safe_probability_fusion(base, retrieval, 0.0), base))
        fused = safe_probability_fusion(base, retrieval, 0.2)
        self.assertTrue(torch.allclose(fused, torch.tensor([0.72, 0.19, 0.09]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
