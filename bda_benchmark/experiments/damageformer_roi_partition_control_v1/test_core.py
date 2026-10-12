import unittest
import torch
from .core import capture, partition, pool, make_heads, Readout, routing_stats, VARIANTS
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map, masked_local_softmax
from bda_benchmark.experiments.damageformer_region_content_v1.core import FullRegionReadout, verify_original_readout


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        ids, valid, gh, gw = candidate_map(8, 8, 4, torch.device('cpu'))
        q = masked_local_softmax(torch.randn(2, 9, 8, 8), valid)
        building = torch.zeros(2, 1, 8, 8)
        building[0, :, :, :4] = 1
        building[1, :, :, 4:] = 1
        self.r = dict(q=q, ids=ids, shape=(8, 8), tokens=torch.zeros(2, gh*gw, 16), building=building)
        self.f = torch.randn(2, 256, 8, 8)

    def test_s0_matches_old_full_readout_forward_and_gradient(self):
        old = FullRegionReadout(16)
        new = Readout(16)
        new.load_state_dict(old.state_dict(), strict=True)
        a = old(self.f, self.r, (16, 16))
        b = new(self.f, partition(self.r, 's0', stride=4), (16, 16))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        (-a[:, 2].log().mean()).backward()
        (-b[:, 2].log().mean()).backward()
        for (name, x), (_, y) in zip(old.named_parameters(), new.named_parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=1e-4, msg=name)

    def test_soft_zero_identity_zeros_preserved_and_normalized(self):
        p = partition(self.r, 's1', strength=0, stride=4)
        self.assertTrue(torch.equal(p['q'], self.r['q']))
        p = partition(self.r, 's1', strength=2, stride=4)
        self.assertTrue(torch.equal(p['q'][self.r['q'] == 0], torch.zeros_like(p['q'][self.r['q'] == 0])))
        torch.testing.assert_close(p['q'].sum(1), torch.ones(2, 64))
        self.assertEqual(p['k'], self.r['tokens'].shape[1])
        self.assertFalse(torch.equal(p['q'], self.r['q']))

    def test_hard_split_pool_and_projection_isolation_batched(self):
        route = partition(self.r, 's2', stride=4)
        group = self.r['building'].flatten(2) >= .5
        self.assertTrue(torch.equal((route['ids'] % 2).bool(), group.expand(-1, 9, -1)))
        h = Readout(16).eval()
        altered = self.f.clone()
        mask = self.r['building'].bool().expand_as(altered)
        altered[~mask] += 100
        # No upsampling: strict isolation at the actual feature/Q resolution.
        a = h(self.f, route, (8, 8))
        b = h(altered, route, (8, 8))
        fg = self.r['building'].bool().expand(-1, 4, -1, -1)
        torch.testing.assert_close(a[fg], b[fg], atol=1e-6, rtol=1e-5)
        self.assertLess(routing_stats(route, self.r['building'])['predicted_binary_mixing_at_05'], 1e-6)

    def test_c2_independent_of_roi_and_same_slot_ceiling(self):
        a = partition(self.r, 'c2', stride=4)
        b = partition(dict(self.r, building=1-self.r['building']), 'c2', stride=4)
        self.assertTrue(torch.equal(a['q'], b['q']))
        self.assertTrue(torch.equal(a['ids'], b['ids']))
        self.assertEqual(a['k'], partition(self.r, 's2', stride=4)['k'])

    def test_heads_equal_initialization_independent_and_frozen_gradients(self):
        heads = make_heads(VARIANTS, 16, 'cpu')
        self.f.requires_grad_(); self.r['q'].requires_grad_(); self.r['building'].requires_grad_()
        for name, h in heads.items():
            for x, y in zip(heads['s0'].parameters(), h.parameters()):
                self.assertTrue(torch.equal(x, y))
                if name != 's0':
                    self.assertNotEqual(x.data_ptr(), y.data_ptr())
            p = h(self.f, partition(self.r, name, stride=4), (16, 16))
            torch.testing.assert_close(p.sum(1), torch.ones(2, 16, 16), atol=1e-6, rtol=1e-5)
            (-p[:, 2].log().mean()).backward()
            for key, parameter in h.named_parameters():
                self.assertIsNotNone(parameter.grad, key)
                self.assertTrue(torch.isfinite(parameter.grad).all(), key)
        self.assertIsNone(self.f.grad)
        self.assertIsNone(self.r['q'].grad)
        self.assertIsNone(self.r['building'].grad)

    def test_all_background_all_building_and_one_location(self):
        for value in (0., 1.):
            for shape, stride in (((1, 1), 8), ((3, 5), 4)):
                h, w = shape
                ids, valid, gh, gw = candidate_map(h, w, stride, torch.device('cpu'))
                r = dict(q=masked_local_softmax(torch.randn(1, 9, h, w), valid),
                         ids=ids, shape=shape, tokens=torch.zeros(1, gh*gw, 16),
                         building=torch.full((1, 1, h, w), value))
                for name in VARIANTS:
                    head = Readout(16)
                    p = head(torch.randn(1, 256, h, w), partition(r, name, stride=stride), shape)
                    (-p[:, 2].log().mean()).backward()
                    self.assertTrue(torch.isfinite(p).all())
                    self.assertTrue(all(torch.isfinite(t.grad).all() for t in head.parameters()))

    def test_real_front_exact_capture_and_immutable_buffers(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        model = SRACSwin(dim=16).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.named_buffers()}
        native = []
        handle = model.base.clf_1.register_forward_hook(lambda m, i, o: native.append(o))
        try:
            r = capture(model, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        finally:
            handle.remove()
        torch.testing.assert_close(r['building'], native[0].float().softmax(1)[:, 1:2])
        self.assertLessEqual(verify_original_readout(model, r, (64, 64)), 1e-5)
        for name, head in make_heads(VARIANTS, 16, 'cpu').items():
            p = head(r['features']['roi'], partition(r, name, stride=8), (64, 64))
            (-p[:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for k, v in model.named_buffers():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_real_objective_optimizer_and_checkpoint_roundtrip(self):
        import io
        from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective
        gt = torch.randint(0, 4, (2, 16, 16))
        gt[:, 0] = 255
        for name, head in make_heads(VARIANTS, 16, 'cpu').items():
            routing = partition(self.r, name, stride=4)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.005)
            before = head.classifier.weight.detach().clone()
            loss = objective(head(self.f, routing, (16, 16)), gt)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            self.assertFalse(torch.equal(before, head.classifier.weight))
            stream = io.BytesIO()
            torch.save(head.state_dict(), stream); stream.seek(0)
            copy = Readout(16)
            copy.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(head(self.f, routing, (16, 16)), copy(self.f, routing, (16, 16)))


if __name__ == '__main__':
    unittest.main()
