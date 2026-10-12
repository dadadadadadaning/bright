# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.RM5_CrossEventMetric import (
    CrossEventResidualMetric,
    cross_event_supervised_contrastive_loss,
    embedding_preservation_loss,
)
from script.standard_ML.train_RM5_cross_event_metric import (
    class_event_index,
    sample_class_event_batch,
)


class RM5CrossEventMetricTests(unittest.TestCase):
    def test_preservation_loss_is_zero_only_for_unchanged_embedding(self):
        original = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(
            float(embedding_preservation_loss(original, original)), 0.0
        )
        changed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        self.assertGreater(
            float(embedding_preservation_loss(changed, original)), 0.0
        )

    def test_step_zero_is_exact_identity_and_adapter_receives_gradient(self):
        torch.manual_seed(7)
        feature = torch.randn(12, 8)
        labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        events = torch.tensor([0, 0, 1, 1] * 3)
        model = CrossEventResidualMetric(8, hidden_dim=6)
        expected = torch.nn.functional.normalize(feature, dim=1)
        torch.testing.assert_close(model(feature), expected, atol=1e-7, rtol=0.0)
        loss = cross_event_supervised_contrastive_loss(
            model(feature), labels, events, temperature=0.1
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.adapter[-1].weight.grad.abs().sum()), 0.0)

    def test_sampler_contains_every_class_and_multiple_events(self):
        labels = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
        codes = torch.tensor(([0] * 4 + [1] * 4) * 3)
        pools = class_event_index(labels, codes)
        batch = sample_class_event_batch(
            pools,
            events_per_class=2,
            samples_per_event=2,
            generator=torch.Generator().manual_seed(3407),
        )
        self.assertEqual(torch.bincount(labels[batch], minlength=3).tolist(), [4, 4, 4])
        for cls in range(3):
            self.assertEqual(torch.unique(codes[batch][labels[batch] == cls]).numel(), 2)

    def test_same_class_same_event_pairs_do_not_create_false_positives(self):
        embedding = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [1.0, 0.0], [0.9, -0.1],
            [-1.0, 0.0], [-0.9, 0.1],
        ])
        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        events = torch.tensor([0, 0, 1, 1, 0, 1])
        loss = cross_event_supervised_contrastive_loss(
            embedding, labels, events, temperature=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
