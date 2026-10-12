import inspect
import unittest

import torch

from bda_benchmark.experiments.damageformer_srac_transformer_v1.build_neighbors import (
    select_ids as parent_select_ids,
)

from .metric import (
    RetrievalProjector,
    balanced_anchor_indices,
    cross_event_contrastive_loss,
    gather_support,
    select_ids,
)


class QueryMetricContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    @staticmethod
    def bank(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        count = len(tokens)
        return {
            "tokens": tokens.clone(),
            "distribution": torch.eye(4)[torch.arange(count) % 4],
            "geometry": torch.arange(count * 5).reshape(count, 5).float(),
            "events": torch.arange(count) % 3,
            "images": torch.arange(count),
            "disasters": torch.arange(count) % 8,
        }

    def test_identity_projector_and_parent_cosine_parity(self):
        tokens = torch.randn(13, 8)
        bank = self.bank(tokens)
        query = torch.randn(5, 8)
        expected = parent_select_ids(
            query,
            bank,
            event=1,
            image=99,
            topk=7,
            policy="global",
            exclusion="cross_event",
        )
        torch.testing.assert_close(
            select_ids(query, bank, 1, 99, 7), expected, rtol=0, atol=0
        )
        projector = RetrievalProjector(8)
        torch.testing.assert_close(
            select_ids(query, bank, 1, 99, 7, projector),
            expected,
            rtol=0,
            atol=0,
        )
        self.assertIsNone(projector.linear.bias)
        torch.testing.assert_close(projector.linear.weight, torch.eye(8))

    def test_selection_excludes_event_and_image_and_pads(self):
        bank = self.bank(torch.eye(4))
        bank["events"] = torch.tensor([7, 8, 9, 10])
        bank["images"] = torch.tensor([20, 21, 22, 23])
        # Closest support is same-event, second closest is same-image.
        query = torch.tensor([[1.0, 1.0, 0.1, 0.0]])
        ids = select_ids(query, bank, event=7, image=21, topk=4)
        self.assertNotIn(0, ids[0].tolist())
        self.assertNotIn(1, ids[0].tolist())
        self.assertEqual(ids[0, 2:].tolist(), [-1, -1])
        signature = inspect.signature(select_ids)
        self.assertEqual(
            list(signature.parameters),
            ["query", "bank", "event", "image", "topk", "projector"],
        )
        self.assertTrue(
            all(
                word not in signature.parameters
                for word in ("gt", "target", "label", "class_id", "policy")
            )
        )

    def test_gather_preserves_original_parent_payload(self):
        bank = self.bank(torch.randn(6, 4, dtype=torch.float64))
        projector = RetrievalProjector(4)
        with torch.no_grad():
            projector.linear.weight.copy_(torch.tensor([
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]))
        ids = select_ids(torch.randn(2, 4), bank, -1, -1, 3, projector)
        support, distribution, valid, metadata = gather_support(ids, bank)
        safe = ids.clamp_min(0)
        torch.testing.assert_close(support, bank["tokens"][safe].float())
        torch.testing.assert_close(distribution, bank["distribution"][safe].float())
        torch.testing.assert_close(metadata["geometry"], bank["geometry"][safe])
        torch.testing.assert_close(metadata["events"], bank["events"][safe])
        torch.testing.assert_close(metadata["disasters"], bank["disasters"][safe])
        self.assertTrue(valid.all())

    def test_no_cross_event_positive_is_finite_backward_safe_zero(self):
        projector = RetrievalProjector(6)
        tokens = torch.randn(5, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2])
        events = torch.zeros(5, dtype=torch.long)
        images = torch.arange(5)
        loss, stats = cross_event_contrastive_loss(
            tokens, labels, events, images, projector
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNone(tokens.grad)
        self.assertIsNotNone(projector.linear.weight.grad)
        self.assertEqual(float(projector.linear.weight.grad.abs().sum()), 0.0)
        self.assertEqual(stats["used_anchor_count"], 0)
        self.assertGreater(stats["skipped_no_positive"], 0)
        self.assertTrue(stats["zero_loss"])

    def test_class_imbalanced_anchor_sampling_is_balanced_and_multi_event(self):
        labels = torch.tensor([0] * 8 + [1] * 2 + [2])
        # Class 2 has no cross-event positive and must be logged/skipped.
        events = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2])
        images = torch.arange(len(labels))
        ids, stats = balanced_anchor_indices(
            labels,
            events,
            images,
            per_class=5,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertEqual(torch.bincount(labels[ids], minlength=4).tolist(), [5, 5, 0, 0])
        self.assertTrue(stats["replacement_by_class"][1])
        self.assertIn(2, stats["classes_without_usable_anchor"])
        self.assertIn(3, stats["missing_class_ids"])
        for anchor in ids.tolist():
            positive = (labels == labels[anchor]) & (events != events[anchor])
            self.assertTrue(bool(positive.any()))

    def test_loss_uses_strict_pairs_high_purity_and_gradients_only_projector(self):
        projector = RetrievalProjector(5)
        tokens = torch.randn(8, 5, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        events = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
        images = torch.arange(8)
        purity = torch.tensor([0.99, 0.98, 0.99, 0.98, 0.97, 0.96, 0.97, 0.20])
        anchors = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        loss, stats = cross_event_contrastive_loss(
            tokens,
            labels,
            events,
            images,
            projector,
            anchor_indices=anchors,
            purity=purity,
            minimum_purity=0.9,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNone(tokens.grad)
        self.assertIsNotNone(projector.linear.weight.grad)
        self.assertGreater(float(projector.linear.weight.grad.abs().sum()), 0.0)
        self.assertEqual(stats["eligible_count"], 7)
        self.assertEqual(stats["skipped_selected_low_purity"], 1)
        self.assertGreater(stats["positive_pair_count"], 0)
        self.assertGreater(stats["negative_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
