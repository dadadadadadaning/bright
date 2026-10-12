# -*- coding: utf-8 -*-
import unittest

import torch

from model.DamageFormer_S9_R1_CEAM import (
    CEAMConfig,
    ChangeLocalAngularMetric,
    ceam_training_loss,
    cross_event_supervised_contrastive_loss,
)
from script.standard_ML.eval_S9_R1_CEAM import validation_gate
from script.standard_ML.s9_r1_ceam_common import cross_event_knn
from script.standard_ML.train_S9_R1_CEAM import EventBalancedBatchSampler


class S9R1CEAMTests(unittest.TestCase):
    def test_forward_arcface_loss_and_gradients(self):
        torch.manual_seed(7)
        model = ChangeLocalAngularMetric(
            CEAMConfig(raw_dim=8, embed_dim=8, hidden_dim=16)
        )
        change = torch.randn(12, 8, requires_grad=True)
        local = torch.randn(12, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2] * 4)
        events = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        output = model({"change_global": change, "local_summary": local})
        self.assertTrue(
            torch.allclose(output["embedding"].norm(dim=1), torch.ones(12), atol=1e-5)
        )
        loss, stats = ceam_training_loss(model, output, labels, events)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats["supcon_valid_anchor_ratio"], 0.0)
        self.assertIsNotNone(model.class_centers.grad)
        self.assertGreater(float(change.grad.abs().sum()), 0.0)
        self.assertGreater(float(local.grad.abs().sum()), 0.0)

    def test_cross_event_supcon_needs_different_event_positive(self):
        embedding = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        same_events = torch.tensor([0, 0, 1, 1, 2, 2])
        loss, stats = cross_event_supervised_contrastive_loss(
            embedding, labels, same_events, temperature=0.1
        )
        self.assertEqual(stats["valid_anchor_ratio"], 0.0)
        self.assertEqual(float(loss), 0.0)
        cross_events = torch.tensor([0, 1, 0, 1, 0, 1])
        loss, stats = cross_event_supervised_contrastive_loss(
            embedding, labels, cross_events, temperature=0.1
        )
        self.assertGreater(stats["valid_anchor_ratio"], 0.0)
        self.assertGreater(float(loss), 0.0)

    def test_event_balanced_sampler_contains_every_class_and_multiple_events(self):
        labels = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
        events = torch.tensor(([0] * 4 + [1] * 4) * 3)
        sampler = EventBalancedBatchSampler(
            torch.arange(24), labels, events,
            events_per_class=2, samples_per_event=2,
            steps_per_epoch=1, seed=5,
        )
        batch = torch.tensor(next(iter(sampler)))
        self.assertEqual(set(labels[batch].tolist()), {0, 1, 2})
        for class_index in range(3):
            self.assertEqual(len(set(events[batch][labels[batch] == class_index].tolist())), 2)

    def test_cross_event_knn_excludes_query_event(self):
        reference = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99], [-1.0, 0.0], [-0.99, 0.01]]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        events = ["same", "other", "other", "third", "other", "third"]
        result = cross_event_knn(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            ["same"],
            reference,
            labels,
            events,
            torch.device("cpu"),
            topk=2,
            max_neighbors_per_event=2,
        )
        chosen = result["neighbor_indices"][0].tolist()
        self.assertNotIn(0, chosen)
        self.assertEqual(int(result["prediction"][0]), 0)

    def test_validation_gate_requires_metric_to_beat_base_and_positive_rescue(self):
        def result(base_macro, knn_macro, damaged, destroyed, net, precision, geometry):
            return {
                "base": {"macro_accuracy": base_macro, "per_class_accuracy": [0.95, 0.60, 0.80]},
                "knn": {"macro_accuracy": knn_macro, "per_class_accuracy": [0.94, damaged, destroyed]},
                "knn_diagnostics": {"positive_geometry_margin_ratio": geometry},
                "selective_intact_to_damage": {"net_rescue": net, "selective_precision": precision},
            }
        good = result(0.78, 0.80, 0.65, 0.81, 5, 0.70, 0.80)
        gate = validation_gate(good, good, good, 0.70, 0.50, 0.02)
        self.assertTrue(gate["accepted_for_memory_rebuild"])
        bad = result(0.78, 0.70, 0.55, 0.70, -2, 0.20, 0.50)
        gate = validation_gate(bad, bad, bad, 0.70, 0.50, 0.02)
        self.assertFalse(gate["accepted_for_memory_rebuild"])


if __name__ == "__main__":
    unittest.main()
