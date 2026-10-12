import copy
import unittest
import torch
from .core import capture, BoundedFeatureModulation
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import MatchedHead


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(2)

    def test_capture_identity_and_frozen_head_backward(self):
        model = SRACSwin(dim=16, stride=4).eval().requires_grad_(False)
        before = {k: x.clone() for k, x in model.named_buffers()}
        r = capture(model, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        self.assertLessEqual(r['identity_max_abs'], 1e-5)
        proto = MatchedHead(dim=16)
        for source, feature in r['features'].items():
            for kind in ('pixel', 'region'):
                head = copy.deepcopy(proto)
                prob = head(feature, r, kind, (64, 64))
                self.assertTrue(torch.allclose(prob.sum(1), torch.ones(1, 64, 64), atol=1e-5))
                (-prob[:, 2].clamp_min(1e-8).log().mean()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.named_buffers()))

    def test_optional_modulation_identity_gradient(self):
        layer = BoundedFeatureModulation(channels=8)
        f = torch.randn(2, 8, 8, 8, requires_grad=True)
        b = torch.rand(2, 1, 4, 4, requires_grad=True)
        out = layer(f, b)
        self.assertTrue(torch.equal(out, f))
        out.square().mean().backward()
        self.assertIsNone(b.grad)
        self.assertGreater(layer.net[-1].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
