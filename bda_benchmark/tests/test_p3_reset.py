import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from bda_benchmark.model.p3_reset import P3ResetModel
from bda_benchmark.model.p3_reset.fusion import ModalitySpecificMoELite, SafeDifferenceResidual
from bda_benchmark.model.p3_reset.heads import ordinal_expected_severity_loss
from bda_benchmark.model.p3_reset.p3_feature_audit import FrozenFeatureReadout, pool_semantic_regions
from bda_benchmark.model.p3_reset.sar_sources.mmov_sar_dino import _normalise_checkpoint
from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import P3PseudoSARSource


class DummyP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Conv2d(3, 256, 1)
        self.sar = nn.Conv2d(3, 256, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)

    def forward(self, pre, post, return_aux=False):
        r = self.rgb(pre); s = self.sar(post); fused = 0.5 * (r + s)
        loc = torch.cat((r[:, :1], r[:, 1:2]), 1)
        dam = self.clf_2(fused)
        if return_aux:
            return loc, dam, {"rgb_feature": r, "sar_feature": s, "fused_roi": fused, "p_building": loc.softmax(1)[:, 1:2]}
        return loc, dam


class DummySAR(nn.Module):
    def forward(self, sar):
        return {"stage4": sar[:, :1].repeat(1, 8, 1, 1)}


class P3ResetTests(unittest.TestCase):
    def inputs(self):
        return torch.randn(2, 3, 16, 16), torch.rand(2, 2, 16, 16) * 255

    def test_c0_is_exact_p3(self):
        p3 = DummyP3(); rgb, sar = self.inputs()
        model = P3ResetModel(p3, "c0")
        _, expected = p3(rgb, P3PseudoSARSource.to_p3_input(sar))
        _, observed = model(rgb, sar)
        self.assertTrue(torch.equal(observed, expected))

    def test_c1_safe_state_is_exact_and_receives_gradient(self):
        rgb, sar = self.inputs(); model = P3ResetModel(DummyP3(), "c1")
        _, base = P3ResetModel(model.p3, "c0")(rgb, sar)
        _, output = model(rgb, sar)
        self.assertTrue(torch.equal(output, base))
        output.square().mean().backward()
        self.assertGreater(float(model.fusion_residual.body[-1].weight.grad.abs().sum()), 0.0)

    def test_c2_safe_state_is_exact_and_receives_gradient(self):
        module = ModalitySpecificMoELite(4, 5, width=8, out_channels=6)
        residual, _ = module(torch.randn(2, 4, 8, 8), torch.randn(2, 5, 4, 4))
        self.assertEqual(float(residual.abs().max()), 0.0)
        residual.sum().backward()
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0.0)

    def test_c1s_is_exact_at_initialisation_and_preserves_occurrence(self):
        rgb, sar = self.inputs(); model = P3ResetModel(DummyP3(), "c1s")
        _, base = P3ResetModel(model.p3, "c0")(rgb, sar)
        _, initial = model(rgb, sar)
        self.assertTrue(torch.equal(initial, base))
        initial.square().mean().backward()
        self.assertGreater(float(model.fusion_residual.body[-1].weight.grad.abs().sum()), 0.0)

        with torch.no_grad():
            model.fusion_residual.body[-1].weight.normal_(std=0.02)
            _, changed = model(rgb, sar)
        base_occurrence = torch.logsumexp(base[:, 2:4], 1) - base[:, 1]
        changed_occurrence = torch.logsumexp(changed[:, 2:4], 1) - changed[:, 1]
        self.assertTrue(torch.allclose(changed_occurrence, base_occurrence, atol=1e-6, rtol=1e-6))
        self.assertGreater(float(((changed[:, 3] - changed[:, 2]) - (base[:, 3] - base[:, 2])).abs().sum()), 0.0)

    def test_region_pooling_uses_building_classes_only(self):
        feature = torch.ones(1, 4, 4, 4)
        damage = torch.zeros(1, 8, 8, dtype=torch.long)
        damage[:, :4, :4] = 1; damage[:, 4:, 4:] = 2
        loc = (damage > 0).long()
        result = pool_semantic_regions(feature, loc, damage, min_area=4)
        self.assertEqual(sorted(result.labels.tolist()), [0, 1])
        self.assertEqual(result.features.shape, (2, 4))

    def test_ordinal_loss_ignores_background(self):
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        self.assertEqual(float(ordinal_expected_severity_loss(logits, target)), 0.0)

    def test_checkpoint_prefixes_are_normalised(self):
        value = torch.ones(1)
        result = _normalise_checkpoint({"teacher": {"module.backbone.cls_token": value}})
        self.assertIn("cls_token", result)

    def test_linear_readout_contract(self):
        head = FrozenFeatureReadout(16)
        self.assertEqual(head(torch.randn(5, 16)).shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
