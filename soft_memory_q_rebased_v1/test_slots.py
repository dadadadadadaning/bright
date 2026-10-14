"""Synthetic routing contracts; no BRIGHT, CUDA or historical replay claim."""
from __future__ import annotations

import unittest

import torch

try:
    from .slots import prepare_slots, route_from_q, slot_labels, slot_pool, sparse_unpool
except ImportError:
    from slots import prepare_slots, route_from_q, slot_labels, slot_pool, sparse_unpool


class SlotContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.q = torch.zeros(1, 9, 4)
        self.q[:, 0] = 1
        self.ids = torch.zeros(9, 4, dtype=torch.long)
        self.b = torch.tensor([[[[0.0, 0.25, 0.75, 1.0]]]])

    def route(self, mode="soft"):
        return route_from_q(self.q, self.ids, 1, (1, 4), self.b, mode)

    def test_conservation_and_ids(self):
        q, ids, count = self.route()
        self.assertEqual(count, 2)
        torch.testing.assert_close(q[:, :9] + q[:, 9:], self.q)
        torch.testing.assert_close(q.sum(1), torch.ones(1, 4))
        self.assertTrue(torch.equal(ids[:, :9], 2 * self.ids[None]))
        self.assertTrue(torch.equal(ids[:, 9:], 2 * self.ids[None] + 1))

    def test_capacity_independent_of_building(self):
        q = torch.zeros(1, 9, 8); q[:, 0] = 1
        ids = torch.zeros(9, 8, dtype=torch.long)
        a = route_from_q(q, ids, 1, (1, 8), torch.zeros(1, 1, 1, 8), "capacity_control")
        b = route_from_q(q, ids, 1, (1, 8), torch.ones(1, 1, 1, 8), "capacity_control")
        self.assertTrue(torch.equal(a[0], b[0]))
        torch.testing.assert_close(a[0][0, 0], torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]))

    def test_feature_and_labels_use_same_mass(self):
        q, ids, count = self.route()
        feature = torch.tensor([[[[0., 1., 2., 3.]]]])
        mean, mass = slot_pool(feature, q, ids, count)
        torch.testing.assert_close(mass, torch.tensor([[[2., 2.]]]))
        torch.testing.assert_close(mean[..., 0], torch.tensor([[[0.625, 2.375]]]))
        target = torch.tensor([[[0, 1, 2, 3]]])
        labels, valid_mass = slot_labels(target, q, ids, (1, 4), count)
        torch.testing.assert_close(mass, valid_mass)
        torch.testing.assert_close(labels[0, 0, 0], torch.tensor([.5, .375, .125, 0.]))
        torch.testing.assert_close(labels[0, 0, 1], torch.tensor([0., .125, .375, .5]))

    def test_background_role_can_have_damaged_label(self):
        self.b.zero_()
        q, ids, count = self.route()
        labels, mass = slot_labels(torch.full((1, 1, 4), 2), q, ids, (1, 4), count)
        self.assertEqual(labels[0, 0, 0, 2].item(), 1.0)
        self.assertEqual(mass[0, 0, 1].item(), 0.0)
        self.assertEqual(labels[0, 0, 1].sum().item(), 0.0)

    def test_positive_tiny_mass_not_clamped(self):
        self.b.fill_(1e-12)
        q, ids, count = self.route()
        mean, mass = slot_pool(torch.full((1, 3, 1, 4), 7.0), q, ids, count)
        torch.testing.assert_close(mean, torch.full((1, 1, 2, 3), 7.0))
        self.assertGreater(mass[0, 0, 1].item(), 0)
        self.assertLess(mass[0, 0, 1].item(), 1e-6)

    def test_empty_slot_mean_and_readback(self):
        self.b.zero_()
        q, ids, count = self.route()
        mean, mass = slot_pool(torch.full((1, 1, 1, 4), 3.0), q, ids, count)
        self.assertEqual(mean[0, 0, 1, 0].item(), 0)
        self.assertEqual(mass[0, 0, 1].item(), 0)
        read = sparse_unpool(mean.reshape(1, 2, 1), q, ids)
        torch.testing.assert_close(read, torch.full((1, 1, 4), 3.0))

    def test_readback_preserves_pixel_role_difference(self):
        q, ids, _ = self.route()
        read = sparse_unpool(torch.tensor([[[2.], [10.]]]), q, ids)
        torch.testing.assert_close(read, torch.tensor([[[2., 4., 8., 10.]]]))

    def test_ignore_area_correct_denominator(self):
        # A full 2x2 source pixel block is one grid cell. Its valid class-2
        # mass is 1/4, not one class-2 pixel plus three fake background pixels.
        q = torch.zeros(1, 9, 1); q[:, 0] = 1
        ids = torch.zeros(9, 1, dtype=torch.long)
        route_q, route_ids, count = route_from_q(q, ids, 1, (1, 1), torch.full((1, 1, 2, 2), .25), "soft")
        labels, mass = slot_labels(torch.tensor([[[2, 255], [255, 255]]]), route_q, route_ids, (1, 1), count)
        torch.testing.assert_close(mass, torch.tensor([[[.1875, .0625]]]))
        self.assertTrue(torch.equal(labels[..., 2], torch.ones(1, 1, 2)))
        self.assertEqual(labels[..., 0].sum().item(), 0.)

    def test_all_ignored_zero_labels_mass(self):
        q, ids, count = self.route()
        labels, mass = slot_labels(torch.full((1, 1, 4), 255), q, ids, (1, 4), count)
        self.assertEqual(labels.sum().item(), 0)
        self.assertEqual(mass.sum().item(), 0)

    def test_invalid_ids_rejected_even_zero_weight(self):
        self.ids[8, 3] = 1
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.route()

    def test_invalid_q_building_label_rejected(self):
        self.q[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): self.route()
        self.q[0, 0, 0] = 1
        self.b[0, 0, 0, 0] = 1.1
        with self.assertRaises(ValueError): self.route()
        self.b[0, 0, 0, 0] = 0
        q, ids, count = self.route()
        with self.assertRaises(ValueError):
            slot_labels(torch.tensor([[[0, 1, 4, 255]]]), q, ids, (1, 4), count)

    def test_preparation_and_batched_ids(self):
        prepared = prepare_slots(torch.randn(1, 256, 1, 4), self.q, self.ids[None], 1, (1, 4), self.b, "soft")
        torch.testing.assert_close(prepared["slot_fraction"].sum(-1), torch.ones(1, 1))
        self.assertTrue(torch.equal(prepared["slot_valid"], prepared["slot_mass"] > 0))
        self.assertEqual(prepared["slot_f"].shape, (1, 1, 2, 256))

    def test_feature_gradients_and_frozen_q(self):
        feature = torch.randn(1, 3, 1, 4, requires_grad=True)
        self.q.requires_grad_(True); self.b.requires_grad_(True)
        q, ids, count = self.route()
        means, _ = slot_pool(feature, q, ids, count)
        means.square().sum().backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertIsNone(self.q.grad)
        self.assertIsNone(self.b.grad)


if __name__ == "__main__":
    unittest.main()
