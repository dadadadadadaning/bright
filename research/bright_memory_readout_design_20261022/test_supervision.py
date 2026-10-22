import math
import unittest
import torch

from supervision import read_supervision, query_slot_targets


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_loss_and_gradient_follow_label_reading_not_target(self):
        scores = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], requires_grad=True)
        labels = torch.tensor([[[1., 0, 0, 0], [0, 0, 1., 0]]])
        p = torch.einsum("brsk,bkc->brsc", scores.softmax(-1), labels)
        target = torch.tensor([[[[0., 0, 1., 0], [1., 0, 0, 0]]]], requires_grad=True)
        loss, stats = read_supervision(p, target, torch.ones(1, 1, 2, dtype=torch.bool),
                                      torch.ones(1, 1, 2))
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        loss.backward()
        self.assertGreater(float(scores.grad[0, 0, 0, 0]), 0)
        self.assertLess(float(scores.grad[0, 0, 0, 1]), 0)
        self.assertIsNone(target.grad)
        self.assertEqual(stats["valid_query_slots"], 2)

    def test_no_gt_and_no_candidates_are_finite_zero(self):
        p = torch.zeros(1, 2, 2, 4, requires_grad=True)
        target = torch.zeros_like(p)
        loss, stats = read_supervision(p, target, torch.zeros(1, 2, 2, dtype=torch.bool),
                                      torch.zeros(1, 2, 2))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(p.grad.abs().sum()), 0.0)
        self.assertTrue(stats["zero_valid_slots"])

    def test_missing_class_is_reported_not_dropped(self):
        p = torch.tensor([[[[1., 0, 0, 0], [1., 0, 0, 0]]]], requires_grad=True)
        t = torch.tensor([[[[0., 0, 1., 0], [0., 0, 1., 0]]]])
        valid = torch.ones(1, 1, 2, dtype=torch.bool)
        loss, stats = read_supervision(p, t, valid, torch.ones(1, 1, 2), p)
        self.assertGreater(float(loss), 18)
        self.assertEqual(stats["mean_target_mass_on_absent_candidate_classes"], 1.0)

    def test_rejects_invalid_probabilities_and_split(self):
        p = torch.ones(1, 1, 2, 4)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            read_supervision(p, p / 4, torch.ones(1, 1, 2, dtype=torch.bool), torch.ones(1, 1, 2))
        with self.assertRaisesRegex(ValueError, "Train-only"):
            query_slot_targets(None, None, None, split="test")

    def test_loss_floor_does_not_modify_input(self):
        p = torch.tensor([[[[1., 0, 0, 0], [1., 0, 0, 0]]]], requires_grad=True)
        old = p.detach().clone()
        read_supervision(p, p.detach(), torch.ones(1, 1, 2, dtype=torch.bool), torch.ones(1, 1, 2))
        self.assertTrue(torch.equal(old, p.detach()))


if __name__ == "__main__":
    unittest.main()
