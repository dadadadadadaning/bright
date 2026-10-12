"""CPU contracts for finite, equal-budget gathering and frozen gradient paths."""
import unittest

import torch

from bda_benchmark.experiments.damageformer_partition_learning_v1.exchange import (
    exchange_pool_route,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    forward_separate, route_from_q,
)


def example():
    torch.manual_seed(18)
    h, w, k = 3, 4, 3
    ids = torch.arange(9).remainder(k)[:, None].expand(9, h*w).clone()
    q = torch.rand(2, 9, h*w)
    q[:, 3:] = 0  # Invalid/duplicate supports stay exactly zero.
    q = (q/q.sum(1, keepdim=True)).requires_grad_()
    building = torch.tensor([0., 1., 0., 1.]*3).reshape(1, 1, h, w).expand(
        2, -1, -1, -1).clone().requires_grad_()
    reference = dict(q=q, ids=ids, shape=(h, w), building=building,
                     tokens=torch.zeros(2, k, 8))
    feature = torch.randn(2, 8, h, w, requires_grad=True)
    return reference, feature


class ExchangeContracts(unittest.TestCase):
    def test_same_budget_support_and_nontrivial_sar_selection(self):
        reference, sar = example()
        uniform, du = exchange_pool_route(reference, sar, .05, mode='uniform')
        guided, ds = exchange_pool_route(reference, sar, .05, mode='sar')
        torch.testing.assert_close(uniform['ids'], guided['ids'])
        self.assertTrue(torch.equal(uniform['q'] > 0, guided['q'] > 0))
        group = (reference['building'].flatten(2) >= .5).long()
        opposite = guided['ids'] % 2 != group
        for route in (uniform, guided):
            torch.testing.assert_close(route['q'].sum(1), torch.ones(2, 12))
            torch.testing.assert_close((route['q']*opposite).sum(1),
                                       torch.full((2, 12), .05))
        self.assertGreater(float((uniform['q']-guided['q']).abs().max()), 1e-5)
        self.assertEqual(du['no_opposite_support_fraction'], ds['no_opposite_support_fraction'])
        self.assertLess(ds['exchange_budget_max_abs_error'], 1e-6)

    def test_all_background_and_zero_feature_fall_back_exactly(self):
        reference, sar = example()
        reference['building'] = torch.zeros_like(reference['building'])
        hard = route_from_q(reference['q'], reference['ids'], 3,
                            reference['shape'], reference['building'])
        for mode in ('uniform', 'sar'):
            route, diag = exchange_pool_route(reference, torch.zeros_like(sar), .1,
                                              mode=mode)
            self.assertTrue(torch.equal(hard['q'], route['q']))
            self.assertEqual(diag['actual_exchange_mean'], 0.)
            self.assertEqual(diag['no_opposite_support_fraction'], 1.)

    def test_epsilon_zero_and_no_new_empty_slots(self):
        reference, sar = example()
        hard = route_from_q(reference['q'], reference['ids'], 3,
                            reference['shape'], reference['building'])
        for mode in ('uniform', 'sar'):
            route, _ = exchange_pool_route(reference, sar, 0., mode=mode)
            self.assertTrue(torch.equal(route['q'], hard['q']))
        # Slot 2 has no hard mass and must never be populated by exchange.
        reference['q'] = reference['q'].detach().clone()
        reference['q'][:, 2] = 0
        reference['q'] /= reference['q'].sum(1, keepdim=True)
        for mode in ('uniform', 'sar'):
            route, _ = exchange_pool_route(reference, sar, .1, mode=mode)
            self.assertTrue(torch.equal(route['q'][:, 2], torch.zeros(2, 12)))
            self.assertTrue(torch.equal(route['q'][:, 11], torch.zeros(2, 12)))

    def test_partial_support_falls_back_per_pixel_not_per_image(self):
        # Foreground pixel 0 and background pixel 1 share seed 0. Background
        # pixel 2 only reaches seed 1, which has no foreground hard support.
        q = torch.zeros(1, 9, 3)
        q[0, 0, :2] = 1.
        q[0, 1, 2] = 1.
        ids = torch.arange(9).remainder(2)[:, None].expand(9, 3)
        reference = dict(q=q, ids=ids, shape=(1, 3), tokens=torch.zeros(1, 2, 4),
                         building=torch.tensor([[[[1., 0., 0.]]]]))
        for mode in ('uniform', 'sar'):
            route, diag = exchange_pool_route(reference, torch.zeros(1, 4, 1, 3),
                                              .05, mode=mode)
            group = reference['building'].flatten(2).long()
            cross = (route['q']*(route['ids'] % 2 != group)).sum(1)
            torch.testing.assert_close(cross, torch.tensor([[.05, .05, 0.]]))
            self.assertAlmostEqual(diag['no_opposite_support_fraction'], 1/3, places=6)
            self.assertEqual(float(route['q'][0, 1, 2]), 1.)

    def test_fixed_readout_is_finite_and_only_head_has_gradients(self):
        reference, sar = example()
        route, _ = exchange_pool_route(reference, sar, .05, mode='sar')
        hard = route_from_q(reference['q'].detach(), reference['ids'], 3,
                            reference['shape'], reference['building'])
        content = torch.randn(2, 256, 3, 4, requires_grad=True)
        head = Readout(dim=8)
        count = sum(p.numel() for p in head.parameters())
        out = forward_separate(head, content, route, hard, (6, 8))
        prob = out['prob']
        self.assertTrue(torch.isfinite(prob).all())
        torch.testing.assert_close(prob.sum(1), torch.ones(2, 6, 8))
        (-prob[:, 1].clamp_min(1e-8).log().mean()).backward()
        self.assertIsNone(reference['q'].grad)
        self.assertIsNone(reference['building'].grad)
        self.assertIsNone(sar.grad)
        self.assertIsNone(content.grad)
        self.assertFalse(route['q'].requires_grad)
        self.assertEqual(count, sum(p.numel() for p in head.parameters()))
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_validates_arguments_and_coordinates(self):
        reference, sar = example()
        for kwargs in (dict(epsilon=-.1), dict(epsilon=.6),
                       dict(epsilon=.1, threshold=1.),
                       dict(epsilon=.1, temperature=0.),
                       dict(epsilon=.1, mode='unknown')):
            with self.assertRaises(ValueError):
                exchange_pool_route(reference, sar, **kwargs)
        with self.assertRaises(ValueError):
            exchange_pool_route(reference, sar[:, :, :-1], .1)


if __name__ == '__main__':
    unittest.main()
