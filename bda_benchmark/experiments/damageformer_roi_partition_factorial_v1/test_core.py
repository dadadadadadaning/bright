"""Factorial routing contracts, including frozen real-front replay."""
import io
import unittest

import torch

from .core import DEFAULT_VARIANTS, VARIANTS, Readout, capture, make_heads, partition, pool, routing_stats
from bda_benchmark.experiments.damageformer_roi_partition_control_v1 import core as old
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map, masked_local_softmax
from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout


def reference(shape=(7, 10), stride=4, batch=2):
    h, w = shape
    ids, valid, gh, gw = candidate_map(h, w, stride, torch.device('cpu'))
    q = masked_local_softmax(torch.randn(batch, 9, h, w), valid)
    building = torch.zeros(batch, 1, h, w)
    building[..., ::2] = 1
    if batch > 1:
        building[1] = 1 - building[1]
    return dict(q=q, ids=ids, shape=shape, tokens=torch.zeros(batch, gh * gw, 16), building=building)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.r = reference()
        self.f = torch.randn(2, 256, *self.r['shape'])

    def test_variant_contract(self):
        self.assertEqual(DEFAULT_VARIANTS, ('g0', 'g1', 's0', 's2'))
        self.assertEqual(set(VARIANTS), {'g0', 'g1', 's0', 's2', 'h0', 'h1'})

    def test_s0_s2_exact_old_forward_and_gradient_parity(self):
        for mode in ('s0', 's2'):
            with self.subTest(mode=mode):
                before = old.partition(self.r, mode, stride=4)
                after = partition(self.r, mode, stride=4)
                for key in ('q', 'ids'):
                    self.assertTrue(torch.equal(before[key], after[key]), key)
                self.assertEqual(before['k'], after['k'])
                a, b = old.Readout(16), Readout(16)
                b.load_state_dict(a.state_dict(), strict=True)
                x, y = a(self.f, before, (14, 20)), b(self.f, after, (14, 20))
                torch.testing.assert_close(x, y, atol=0, rtol=0)
                (-x[:, 2].log().mean()).backward()
                (-y[:, 2].log().mean()).backward()
                for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)

    def test_grid_manual_block_means_nondivisible_shape(self):
        route = partition(self.r, 'g0', stride=4)
        h, w = self.r['shape']
        values = torch.arange(2 * h * w * 3, dtype=torch.float32).reshape(2, h, w, 3)
        actual, mass = pool(values.reshape(2, h*w, 3), route['q'], route['ids'], route['k'])
        expected, sizes = [], []
        for yy in range(0, h, 4):
            for xx in range(0, w, 4):
                block = values[:, yy:yy+4, xx:xx+4]
                expected.append(block.mean((1, 2)))
                sizes.append(block.shape[1] * block.shape[2])
        torch.testing.assert_close(actual, torch.stack(expected, 1), atol=0, rtol=0)
        torch.testing.assert_close(mass, torch.tensor(sizes, dtype=torch.float32)[None].expand(2, -1))
        self.assertTrue(torch.equal(route['q'][:, 4], torch.ones_like(route['q'][:, 4])))
        self.assertEqual(int(torch.count_nonzero(route['q'])), 2*h*w)

    def test_grid_independent_of_q_and_unsplit_independent_of_building(self):
        h, w = self.r['shape']
        _, valid, _, _ = candidate_map(h, w, 4, torch.device('cpu'))
        altered = dict(self.r, q=masked_local_softmax(torch.randn(2, 9, h, w)*7, valid))
        for mode in ('g0', 'g1'):
            a, b = partition(self.r, mode, stride=4), partition(altered, mode, stride=4)
            self.assertTrue(torch.equal(a['q'], b['q']))
            self.assertTrue(torch.equal(a['ids'], b['ids']))
        for mode in ('g0', 's0', 'h0'):
            a = partition(self.r, mode, stride=4)
            b = partition(dict(self.r, building=1-self.r['building']), mode, stride=4)
            self.assertTrue(torch.equal(a['q'], b['q']))
            self.assertTrue(torch.equal(a['ids'], b['ids']))

    def test_hard_argmax_preserves_original_candidate_ids(self):
        q = self.r['q']
        winner = q.argmax(1, keepdim=True)
        original_ids = self.r['ids'][None].expand(2, -1, -1)
        for mode in ('h0', 'h1'):
            route = partition(self.r, mode, stride=4)
            self.assertTrue(torch.equal(route['q'], torch.zeros_like(q).scatter_(1, winner, 1)))
            chosen = route['ids'].gather(1, winner)
            expected = original_ids.gather(1, winner)
            if mode == 'h1':
                expected = 2 * expected + (self.r['building'].flatten(2) >= .5).long()
            self.assertTrue(torch.equal(chosen, expected))

    def test_all_modes_normalization_valid_mask_and_slot_ceiling(self):
        h, w = self.r['shape']
        _, valid, _, _ = candidate_map(h, w, 4, torch.device('cpu'))
        for mode in VARIANTS:
            route = partition(self.r, mode, stride=4)
            torch.testing.assert_close(route['q'].sum(1), torch.ones(2, h*w))
            self.assertTrue((route['q'] >= 0).all())
            self.assertTrue(torch.isfinite(route['q']).all())
            self.assertEqual(int(torch.count_nonzero(route['q'][:, ~valid])), 0)
            self.assertTrue((route['ids'] >= 0).all())
            self.assertTrue((route['ids'] < route['k']).all())
            self.assertEqual(route['k'], self.r['tokens'].shape[1] * (2 if mode in ('g1', 's2', 'h1') else 1))

    def test_split_isolation_at_native_grid(self):
        # Full-size bilinear upsampling can blend boundary outputs, so this
        # contract intentionally tests the actual routing/feature resolution.
        foreground = self.r['building'].bool()
        altered = self.f.clone()
        altered[~foreground.expand_as(altered)] += 100
        for mode in ('g1', 's2', 'h1'):
            route = partition(self.r, mode, stride=4)
            group = (self.r['building'].flatten(2) >= .5).expand(-1, 9, -1)
            self.assertTrue(torch.equal((route['ids'] % 2).bool(), group))
            head = Readout(16).eval()
            a, b = head(self.f, route, self.r['shape']), head(altered, route, self.r['shape'])
            mask = foreground.expand_as(a)
            torch.testing.assert_close(a[mask], b[mask], atol=1e-6, rtol=1e-5)
            self.assertLess(routing_stats(route, self.r['building'])['predicted_binary_mixing_at_05'], 1e-6)

    def test_equal_independent_heads_and_frozen_inputs(self):
        heads = make_heads(VARIANTS, 16, 'cpu')
        self.f.requires_grad_()
        self.r['q'].requires_grad_()
        self.r['building'].requires_grad_()
        self.r['tokens'].requires_grad_()
        for mode, head in heads.items():
            for a, b in zip(heads['g0'].parameters(), head.parameters()):
                self.assertTrue(torch.equal(a, b))
                if mode != 'g0':
                    self.assertNotEqual(a.data_ptr(), b.data_ptr())
            prob = head(self.f, partition(self.r, mode, stride=4), (14, 20))
            torch.testing.assert_close(prob.sum(1), torch.ones(2, 14, 20), atol=1e-6, rtol=1e-5)
            (-prob[:, 2].log().mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        for value in (self.f, self.r['q'], self.r['building'], self.r['tokens']):
            self.assertIsNone(value.grad)

    def test_empty_groups_one_pixel_and_uniform_building_masks(self):
        for value in (0., 1.):
            for shape, stride in (((1, 1), 8), ((3, 5), 4)):
                r = reference(shape, stride, batch=1)
                r['building'].fill_(value)
                for mode in VARIANTS:
                    with self.subTest(value=value, shape=shape, mode=mode):
                        route = partition(r, mode, stride=stride)
                        f = torch.randn(1, 256, *shape)
                        pooled, mass = pool(f.flatten(2).transpose(1, 2), route['q'], route['ids'], route['k'])
                        self.assertTrue(torch.isfinite(pooled).all())
                        if mode in ('g1', 's2', 'h1'):
                            self.assertGreater(int((mass == 0).sum()), 0)
                            self.assertEqual(int(torch.count_nonzero(pooled[mass == 0])), 0)
                        head = Readout(16)
                        prob = head(f, route, shape)
                        (-prob[:, 2].log().mean()).backward()
                        self.assertTrue(torch.isfinite(prob).all())
                        torch.testing.assert_close(prob.sum(1), torch.ones(1, *shape))
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_real_front_capture_replay_and_frozen_buffers(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        model = SRACSwin(dim=16).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.named_buffers()}
        native = []
        handle = model.base.clf_1.register_forward_hook(lambda m, i, o: native.append(o))
        try:
            r = capture(model, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        finally:
            handle.remove()
        self.assertEqual(len(native), 1)
        torch.testing.assert_close(r['building'], native[0].float().softmax(1)[:, 1:2])
        self.assertEqual(tuple(r['building'].shape[-2:]), tuple(r['shape']))
        self.assertLessEqual(verify_original_readout(model, r, (64, 64)), 1e-5)
        for mode, head in make_heads(VARIANTS, 16, 'cpu').items():
            prob = head(r['features']['roi'], partition(r, mode, stride=8), (64, 64))
            (-prob[:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_real_nll_lovasz_optimizer_update_and_checkpoint(self):
        from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective
        target = torch.randint(0, 4, (2, 14, 20))
        target[:, 0] = 255
        for mode, head in make_heads(VARIANTS, 16, 'cpu').items():
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
            restored.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(head(self.f, route, (14, 20)), restored(self.f, route, (14, 20)))


if __name__ == '__main__':
    unittest.main()
