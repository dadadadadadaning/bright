"""CPU-only routing contracts; never reads dataset or user checkpoints."""
import unittest
import torch
from .routing import route_from_q, forward_separate, valid_read_route, encode_regions, decode_regions, route_diagnostics
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout, partition
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.test_core import reference


class RoutingContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3407)
        self.r = reference()
        self.f = torch.randn(2,256,*self.r['shape'],requires_grad=True)
        self.head = Readout(16).eval()

    def route(self, epsilon):
        r = self.r
        return route_from_q(r['q'],r['ids'],r['tokens'].shape[1],r['shape'],r['building'],epsilon)

    def test_exact_route_and_diagonal_output(self):
        for name,epsilon in (('hard',0.),('exchange',.05)):
            expected = partition(self.r,name,epsilon=epsilon,stride=4)
            route = self.route(epsilon)
            torch.testing.assert_close(route['q'],expected['q'],atol=0,rtol=0)
            self.assertTrue(torch.equal(route['ids'],expected['ids']))
            original = self.head(self.f,expected,(14,20))
            actual = forward_separate(self.head,self.f,route,route,(14,20))['prob']
            torch.testing.assert_close(actual,original,atol=2e-6,rtol=2e-5)

    def test_all_four_routes_finite_and_normalized(self):
        for a in (0.,.05):
            for b in (0.,.05):
                result = forward_separate(self.head,self.f,self.route(a),self.route(b),(14,20))
                self.assertTrue(torch.isfinite(result['prob']).all())
                torch.testing.assert_close(result['prob'].sum(1),torch.ones(2,14,20))

    def test_empty_cross_read_is_blocked_before_resize(self):
        self.r['building'].zero_()
        hard,exchange = self.route(0.),self.route(.05)
        encoded = encode_regions(self.head,self.f,hard)
        route = valid_read_route(exchange,encoded['mass'])
        torch.testing.assert_close(route['q'][:,9:],torch.zeros_like(route['q'][:,9:]))
        logits = self.head.classifier(encoded['tokens'])
        before = decode_regions(logits,exchange,encoded['mass'],(14,20))
        after_logits = logits.clone()
        after_logits[encoded['mass']==0] = torch.tensor([100.,-100.,-100.,100.])
        after = decode_regions(after_logits,exchange,encoded['mass'],(14,20))
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        stats = route_diagnostics(hard,exchange,encoded['mass'],encoded['attention_mass'],self.r['building'])
        self.assertAlmostEqual(stats['blocked_read_mass_fraction'],.05,places=6)
        self.assertEqual(stats['actual_cross_read_fraction'],0.)

    def test_gradient_reaches_q_but_not_building_or_content(self):
        self.r['q'].requires_grad_(); self.r['building'].requires_grad_()
        route = self.route(.05)
        prob = forward_separate(self.head,self.f,route,route,(14,20))['prob']
        (-prob[:,2].log().mean()).backward()
        self.assertIsNone(self.f.grad); self.assertIsNone(self.r['building'].grad)
        self.assertIsNotNone(self.r['q'].grad)
        self.assertGreater(float(self.r['q'].grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.head.parameters()))

    def test_zero_quality_is_not_silent_fallback(self):
        route = self.route(0.)
        with self.assertRaises(ValueError):
            valid_read_route(route,torch.zeros(2,route['k']))

    def test_tiny_positive_mass_not_pruned(self):
        route = self.route(.05)
        mass = torch.full((2,route['k']),1e-10)
        result = valid_read_route(route,mass)
        torch.testing.assert_close(result['q'],route['q'])

    def test_invalid_and_foreign_routes_rejected(self):
        with self.assertRaises(ValueError): self.route(-.1)
        route = self.route(0.)
        changed = dict(route,k=route['k']+1)
        with self.assertRaises(ValueError):
            forward_separate(self.head,self.f,route,changed,(14,20))


if __name__ == '__main__':
    unittest.main()
