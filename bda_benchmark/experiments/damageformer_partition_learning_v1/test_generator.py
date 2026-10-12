"""CPU contracts for generator-only changes, including structural supervision."""
import unittest

import torch
from torch.nn import functional as F

from .generator import dual_assignment, structure_reconstruction_loss
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import candidate_map, iterative_assignment


class GeneratorContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(34)

    def test_dual_beta_zero_and_identical_descriptor_replay(self):
        for h, w in ((1, 1), (3, 5), (6, 8)):
            rgb = torch.randn(2, 5, h, w)
            sar = torch.randn(2, 7, h, w)
            ids, valid, gh, gw = candidate_map(h, w, 2, rgb.device)
            old = iterative_assignment(rgb, ids, valid, torch.zeros(2, 1, h, w), gh*gw, 2)
            q = dual_assignment(rgb, sar, ids, valid, gh*gw, 2, beta=0.)
            torch.testing.assert_close(q, old, rtol=1e-5, atol=1e-6)
            same = dual_assignment(rgb, rgb, ids, valid, gh*gw, 2, beta=.2)
            torch.testing.assert_close(same, old, rtol=1e-5, atol=1e-6)

    def test_dual_gradients_invalid_and_empty_candidates(self):
        rgb = torch.randn(2, 5, 5, 7, requires_grad=True)
        sar = torch.randn(2, 3, 5, 7, requires_grad=True)
        ids, valid, gh, gw = candidate_map(5, 7, 2, rgb.device)
        # Extra allocated empty regions do not attract mass.
        q = dual_assignment(rgb, sar, ids, valid, gh*gw+2, 2)
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q.masked_select(~valid[None].expand_as(q)) == 0).all())
        torch.testing.assert_close(q.sum(1), torch.ones_like(q[:, 0]))
        (q*torch.randn_like(q)).sum().backward()
        for tensor in (rgb, sar):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        batched = dual_assignment(rgb, sar, ids[None].expand(2, -1, -1), valid[None].expand(2, -1, -1), gh*gw+2, 2)
        torch.testing.assert_close(batched, q)

    def test_dual_validation_and_autocast(self):
        rgb = torch.randn(1, 4, 3, 5, requires_grad=True)
        ids, valid, gh, gw = candidate_map(3, 5, 2, rgb.device)
        args = (rgb, rgb, ids, valid, gh*gw, 2)
        for kwargs in ({'beta': -1}, {'beta': float('nan')}, {'temperature': 0}, {'iterations': 0}):
            with self.assertRaises(ValueError):
                dual_assignment(*args, **kwargs)
        invalid = valid.clone()
        invalid[4, 0] = False
        with self.assertRaises(ValueError):
            dual_assignment(rgb, rgb, ids, invalid, gh*gw, 2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            q = dual_assignment(*args)
        self.assertEqual(q.dtype, torch.float32)

    @staticmethod
    def two_pixel_q(mixed=False):
        q = torch.zeros(1, 9, 2)
        ids = torch.zeros(9, 2, dtype=torch.long)
        if mixed:
            q[:, 4] = 1.
        else:
            ids[4, 1] = 1
            q[:, 4] = 1.
        return q, ids

    def test_structure_pure_mixed_and_no_multiclass_interpolation(self):
        gt = torch.tensor([[[0, 3]]])
        q, ids = self.two_pixel_q()
        pure = structure_reconstruction_loss(q, ids, 2, (1, 2), gt)
        self.assertEqual(pure.item(), 0)
        q, ids = self.two_pixel_q(mixed=True)
        mixed = structure_reconstruction_loss(q, ids, 2, (1, 2), gt)
        torch.testing.assert_close(mixed, torch.tensor(2.).log())
        # Intact/Damaged/Destroyed are all foreground, never interpolated IDs.
        all_buildings = torch.tensor([[[1, 2, 3, 1], [2, 3, 1, 2]]])
        self.assertEqual(structure_reconstruction_loss(q, ids, 2, (1, 2), all_buildings).item(), 0)

    def test_structure_ignore_valid_fraction_and_batch_isolation(self):
        q, ids = self.two_pixel_q(mixed=True)
        gt = torch.tensor([[[0, 255, 1, 1], [255, 255, 1, 1]]])
        loss = structure_reconstruction_loss(q, ids, 2, (1, 2), gt)
        # First cell has valid fraction .25, second 1; region foreground=.8.
        expected = F.binary_cross_entropy(torch.tensor(.8), torch.tensor(.8))
        torch.testing.assert_close(loss, expected)
        ignored_second = torch.tensor([[[0, 0, 255, 255], [0, 0, 255, 255]]])
        self.assertEqual(structure_reconstruction_loss(q, ids, 2, (1, 2), ignored_second).item(), 0)
        batch = torch.cat((torch.zeros(1, 1, 2, dtype=torch.long), torch.ones(1, 1, 2, dtype=torch.long)), 0)
        self.assertEqual(structure_reconstruction_loss(q.expand(2, -1, -1), ids, 2, (1, 2), batch).item(), 0)

    def test_structure_gradients_all_ignored_and_positive_tiny_mass(self):
        logits = torch.randn(1, 9, 4, requires_grad=True)
        ids = torch.arange(9)[:, None].expand(9, 4)
        loss = structure_reconstruction_loss(logits.softmax(1), ids, 11, (2, 2), torch.tensor([[[0, 0], [1, 1]]]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        logits2 = torch.randn(1, 9, 4, requires_grad=True)
        ignored = structure_reconstruction_loss(logits2.softmax(1), ids, 11, (2, 2), torch.full((1, 2, 2), 255))
        ignored.backward()
        self.assertEqual(ignored.item(), 0)
        self.assertEqual(logits2.grad.abs().sum().item(), 0)
        # Several tiny positive pure-foreground regions must still represent 1.
        tinyq = torch.zeros(1, 9, 2)
        tinyq[:, 0] = 1.-1e-7
        tinyq[:, 1] = 1e-7
        tinyids = torch.zeros(9, 2, dtype=torch.long)
        tinyids[1] = 1
        tiny = structure_reconstruction_loss(tinyq, tinyids, 3, (1, 2), torch.ones(1, 1, 2, dtype=torch.long))
        self.assertEqual(tiny.item(), 0)

    def test_structure_validation(self):
        q, ids = self.two_pixel_q()
        with self.assertRaises(ValueError):
            structure_reconstruction_loss(q*.5, ids, 2, (1, 2), torch.zeros(1, 1, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            structure_reconstruction_loss(q, ids, 2, (1, 2), torch.tensor([[[0, 7]]]))
        with self.assertRaises(ValueError):
            structure_reconstruction_loss(q, ids, 2, (2, 2), torch.zeros(1, 2, 2, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
