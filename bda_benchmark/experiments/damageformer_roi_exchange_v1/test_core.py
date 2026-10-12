"""CPU contracts for bounded exchange; no dataset or GPU training."""
import io
import unittest

import torch

from .core import Readout, make_heads, partition, pool, routing_stats, verify_hard_parity
from bda_benchmark.experiments.damageformer_roi_partition_control_v1 import core as old
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.test_core import reference
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map


class ExchangeContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.r = reference()
        self.f = torch.randn(2, 256, *self.r['shape'])

    def test_zero_exchange_matches_old_s2_forward_and_every_parameter_gradient(self):
        for mode in ('hard', 'exchange'):
            a, b = old.Readout(16), Readout(16)
            b.load_state_dict(a.state_dict(), strict=True)
            pa = a(self.f, old.partition(self.r, 's2', stride=4), (14, 20))
            pb = b(self.f, partition(self.r, mode, epsilon=0., stride=4), (14, 20))
            torch.testing.assert_close(pa, pb, atol=2e-6, rtol=2e-5)
            (-pa[:, 2].log().mean()).backward()
            (-pb[:, 2].log().mean()).backward()
            for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
                self.assertIsNotNone(x.grad, name)
                self.assertIsNotNone(y.grad, name)
                torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=2e-4, msg=name)

    def test_half_exchange_matches_unsplit_s0(self):
        a, b = old.Readout(16), Readout(16)
        b.load_state_dict(a.state_dict(), strict=True)
        pa = a(self.f, old.partition(self.r, 's0', stride=4), (14, 20))
        pb = b(self.f, partition(self.r, 'exchange', epsilon=.5, stride=4), (14, 20))
        torch.testing.assert_close(pa, pb, atol=2e-6, rtol=2e-5)

    def test_parity_helper_does_not_mutate_head_gradients_or_rng(self):
        head = Readout(16).train()
        for p in head.parameters():
            p.grad = torch.ones_like(p)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = verify_hard_parity(self.f, self.r, head, (14, 20), stride=4)
        self.assertTrue(report['passed'])
        self.assertTrue(head.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for k, value in head.state_dict().items():
            self.assertTrue(torch.equal(value, before[k]), k)
        for p in head.parameters():
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))

    def test_routing_statistics_measure_fixed_cross_group_mass(self):
        for mode, epsilon, expected in (('hard', .05, 0.), ('exchange', .05, .05), ('exchange', .5, .5)):
            route = partition(self.r, mode, epsilon=epsilon, stride=4)
            stats = routing_stats(route, self.r['building'])
            self.assertAlmostEqual(stats['cross_group_weight_fraction'], expected, places=6)
            self.assertEqual(stats['candidates'], 18)
            self.assertEqual(stats['allocated'], 2*self.r['tokens'].shape[1])

    def test_exchange_mass_normalization_valid_candidates_and_no_mutation(self):
        q_before, ids_before = self.r['q'].clone(), self.r['ids'].clone()
        _, valid, _, _ = candidate_map(*self.r['shape'], 4, torch.device('cpu'))
        for epsilon in (0., .05, .5):
            route = partition(self.r, 'exchange', epsilon=epsilon, stride=4)
            self.assertEqual(route['q'].shape, (2, 18, 70))
            self.assertEqual(route['ids'].shape, route['q'].shape)
            self.assertEqual(route['k'], 2*self.r['tokens'].shape[1])
            torch.testing.assert_close(route['q'].sum(1), torch.ones(2, 70))
            torch.testing.assert_close(route['q'][:, :9] + route['q'][:, 9:], self.r['q'])
            self.assertEqual(int(torch.count_nonzero(route['q'][:, ~torch.cat([valid, valid])])), 0)
            self.assertTrue((route['q'] >= 0).all())
            self.assertTrue((route['ids'] >= 0).all())
            self.assertTrue((route['ids'] < route['k']).all())
        self.assertTrue(torch.equal(q_before, self.r['q']))
        self.assertTrue(torch.equal(ids_before, self.r['ids']))

    def test_constant_floor_uses_binary_group_not_confidence(self):
        r = dict(self.r, building=self.r['building']*.02+.49)
        a = partition(self.r, 'exchange', epsilon=.05, stride=4)
        b = partition(r, 'exchange', epsilon=.05, stride=4)
        self.assertTrue(torch.equal(a['q'], b['q']))
        r = dict(self.r, building=torch.full_like(self.r['building'], .5))
        route = partition(r, 'exchange', epsilon=.05, stride=4)
        torch.testing.assert_close(route['q'][:, 9:], self.r['q']*.95)

    def test_native_hard_isolation_and_positive_exchange_sharing(self):
        # Check scalar aggregation itself, avoiding accidental learned-head
        # insensitivity to a changed feature direction.
        values = self.r['building'].flatten(2).transpose(1, 2)
        outputs = {}
        for epsilon in (0., .05):
            route = partition(self.r, 'exchange', epsilon=epsilon, stride=4)
            tokens, _ = pool(values, route['q'], route['ids'], route['k'])
            result = torch.zeros_like(values)
            for j in range(route['q'].shape[1]):
                result += tokens.gather(1, route['ids'][:, j, :, None])*route['q'][:, j, :, None]
            outputs[epsilon] = result
        torch.testing.assert_close(outputs[0.], values, atol=1e-6, rtol=1e-5)
        self.assertTrue((outputs[.05][values == 0] > 0).all())
        self.assertTrue((outputs[.05][values == 1] < 1).all())

    def test_batch_independence(self):
        head = Readout(16).eval()
        batch = head(self.f, partition(self.r, 'exchange', stride=4), (14, 20))
        for i in range(2):
            single = {k: (v[i:i+1] if k in ('q', 'building', 'tokens') else v)
                      for k, v in self.r.items()}
            pred = head(self.f[i:i+1], partition(single, 'exchange', stride=4), (14, 20))
            torch.testing.assert_close(batch[i:i+1], pred, atol=2e-6, rtol=2e-5)

    def test_empty_tokens_and_zero_weight_support_cannot_change_output(self):
        r = reference((3, 5), 4, batch=1)
        r['building'].zero_()
        route = partition(r, 'hard', stride=4)
        f = torch.randn(1, 256, 3, 5)
        head = Readout(16).eval()
        a = head(f, route, (6, 10))
        # Route every zero-weight support to a new unused token. That token
        # still passes through biased embedding/classification but has no mass.
        changed = dict(route, ids=route['ids'].clone(), k=route['k']+1)
        changed['ids'][route['q'] == 0] = route['k']
        b = head(f, changed, (6, 10))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_uniform_groups_one_pixel_and_empty_mass_are_finite(self):
        for shape in ((1, 1), (3, 5)):
            for value in (0., 1.):
                r = reference(shape, 4, batch=1)
                r['building'].fill_(value)
                for epsilon in (0., .05, .5):
                    route = partition(r, 'exchange', epsilon=epsilon, stride=4)
                    f = torch.randn(1, 256, *shape)
                    z, mass = pool(f.flatten(2).transpose(1, 2), route['q'], route['ids'], route['k'])
                    self.assertTrue(torch.isfinite(z).all())
                    self.assertEqual(int(torch.count_nonzero(z[mass == 0])), 0)
                    head = Readout(16)
                    prob = head(f, route, shape)
                    torch.testing.assert_close(prob.sum(1), torch.ones(1, *shape))
                    (-prob[:, 2].log().mean()).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_invalid_epsilon_is_rejected(self):
        for epsilon in (-.01, .5001, float('nan'), float('inf'), -float('inf')):
            for mode in ('hard', 'exchange'):
                with self.subTest(epsilon=epsilon, mode=mode), self.assertRaises(ValueError):
                    partition(self.r, mode, epsilon=epsilon, stride=4)

    def test_equal_independent_head_capacity_and_frozen_inputs(self):
        heads = make_heads(('hard', 'exchange'), 16, 'cpu')
        self.f.requires_grad_()
        for key in ('q', 'building', 'tokens'):
            self.r[key].requires_grad_()
        for a, b in zip(heads['hard'].parameters(), heads['exchange'].parameters()):
            self.assertTrue(torch.equal(a, b))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        for mode, head in heads.items():
            prob = head(self.f, partition(self.r, mode, stride=4), (14, 20))
            (-prob[:, 2].log().mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertIsNone(self.f.grad)
        for key in ('q', 'building', 'tokens'):
            self.assertIsNone(self.r[key].grad)

    def test_real_nll_lovasz_update_and_checkpoint_roundtrip(self):
        from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective
        target = torch.randint(0, 4, (2, 14, 20))
        target[:, 0] = 255
        for mode, head in make_heads(('hard', 'exchange'), 16, 'cpu').items():
            route = partition(self.r, mode, stride=4)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.005)
            before = head.classifier.weight.detach().clone()
            loss = objective(head(self.f, route, (14, 20)), target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            self.assertFalse(torch.equal(before, head.classifier.weight))
            stream = io.BytesIO()
            torch.save(head.state_dict(), stream)
            stream.seek(0)
            restored = Readout(16)
            restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
            torch.testing.assert_close(head(self.f, route, (14, 20)), restored(self.f, route, (14, 20)))

    def test_real_front_native_capture_and_frozen_buffers(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout
        model = SRACSwin(dim=16).eval().requires_grad_(False)
        before = {name: value.clone() for name, value in model.named_buffers()}
        r = old.capture(model, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(r['building'].shape[-2:]), tuple(r['shape']))
        self.assertLessEqual(verify_original_readout(model, r, (64, 64)), 1e-5)
        for mode, head in make_heads(('hard', 'exchange'), 16, 'cpu').items():
            prob = head(r['features']['roi'], partition(r, mode, stride=8), (64, 64))
            (-prob[:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == '__main__':
    unittest.main()
