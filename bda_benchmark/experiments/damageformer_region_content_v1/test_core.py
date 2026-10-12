import unittest
import torch
from .core import FullRegionReadout, make_heads, capture, verify_original_readout
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map, masked_local_softmax,
)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        ids, valid, gh, gw = candidate_map(8, 8, 4, torch.device('cpu'))
        q = masked_local_softmax(torch.randn(2, 9, 8, 8), valid)
        self.ref = dict(q=q, ids=ids, tokens=torch.zeros(2, gh*gw, 16), shape=(8, 8))
        self.f = torch.randn(2, 256, 8, 8)

    def test_identical_initialization_independent_parameters(self):
        heads = make_heads(['normal', 'raw', 'roi'], 16, 'cpu')
        for name in ('raw', 'roi'):
            for a, b in zip(heads['normal'].parameters(), heads[name].parameters()):
                self.assertTrue(torch.equal(a, b))
                self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_probabilities_and_frozen_gradient_boundary(self):
        head = FullRegionReadout(16)
        self.f.requires_grad_()
        self.ref['q'].requires_grad_()
        old_q = self.ref['q'].detach().clone()
        p = head(self.f, self.ref, (16, 16))
        self.assertTrue(torch.allclose(p.sum(1), torch.ones(2, 16, 16), atol=1e-6))
        target = torch.randint(0, 4, (2, 16, 16))
        loss = -p.log().gather(1, target[:, None]).mean()
        loss.backward()
        self.assertIsNone(self.f.grad)
        self.assertIsNone(self.ref['q'].grad)
        self.assertTrue(torch.equal(old_q, self.ref['q']))
        for name, param in head.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for module in (head.embedding.score, head.embedding.project,
                       head.embedding.stats, head.classifier):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)

    def test_no_input_layernorm_and_shared_content_identity(self):
        heads = make_heads(['normal', 'roi'], 16, 'cpu')
        a = heads['normal'](self.f, self.ref, (8, 8))
        b = heads['roi'](self.f, self.ref, (8, 8))
        self.assertTrue(torch.equal(a, b))
        self.assertIsInstance(heads['roi'].embedding.project, torch.nn.Conv2d)

    def test_real_front_capture_replay_and_buffers_unchanged(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        model = SRACSwin(dim=16).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.named_buffers()}
        x, y = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        ref = capture(model, x, y)
        self.assertLessEqual(ref['identity_max_abs'], 1e-5)
        self.assertLessEqual(verify_original_readout(model, ref, (64, 64)), 1e-5)
        heads = make_heads(['normal', 'raw', 'roi'], 16, 'cpu')
        for source, head in heads.items():
            p = head(ref['features'][source], ref, (64, 64))
            (-p[:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, buf in model.named_buffers():
            self.assertTrue(torch.equal(before[name], buf), name)


if __name__ == '__main__':
    unittest.main()
