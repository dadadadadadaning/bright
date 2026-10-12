# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


# Unit tests exercise the new head without constructing two Swin encoders.
fake_base_module = types.ModuleType("model.DamageFormer_S3_BGMP_K3")


class _FakeS3(nn.Module):
    pass


fake_base_module.DamageFormer_S3_BGMP_K3 = _FakeS3
sys.modules.setdefault("model", types.ModuleType("model"))
sys.modules["model.DamageFormer_S3_BGMP_K3"] = fake_base_module

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "DamageFormer_S8_MICB_DSMIL.py"
)
spec = importlib.util.spec_from_file_location("s8_micb_unit", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

BuildingBagSpec = module.BuildingBagSpec
MICBDSMILHead = module.MICBDSMILHead
build_training_bag_specs = module.build_training_bag_specs
connected_components_4 = module.connected_components_4
hierarchical_log_probs = module.hierarchical_log_probs


class S8MICBDSMILTests(unittest.TestCase):
    def test_components_and_class_separation(self):
        loc = torch.zeros(1, 10, 12, dtype=torch.long)
        damage = torch.zeros_like(loc)
        loc[0, 1:5, 1:5] = 1
        damage[0, 1:5, 1:5] = 1
        # Touches the intact block, but the damage label must create another bag.
        loc[0, 1:5, 5:8] = 1
        damage[0, 1:5, 5:8] = 2
        loc[0, 7:9, 8:11] = 1
        damage[0, 7:9, 8:11] = 3
        bags = build_training_bag_specs(loc, damage, min_area=2)
        self.assertEqual([bag.label for bag in bags], [1, 2, 3])
        self.assertEqual([bag.area for bag in bags], [16, 12, 6])
        self.assertEqual(len(connected_components_4(loc[0] > 0)), 2)

    def test_hierarchical_probabilities_normalize(self):
        exist = torch.tensor([[4.0, -1.0], [-2.0, 3.0], [-2.0, 3.0]])
        severity = torch.tensor([[0.0, 0.0], [3.0, -2.0], [-2.0, 3.0]])
        log_prob = hierarchical_log_probs(exist, severity)
        torch.testing.assert_close(log_prob.exp().sum(dim=1), torch.ones(3))
        self.assertEqual(log_prob.argmax(dim=1).tolist(), [0, 1, 2])

    def test_multimodal_bag_forward_and_gradient(self):
        torch.manual_seed(7)
        head = MICBDSMILHead(
            in_channels=16,
            token_dim=16,
            topk=3,
            max_tokens=16,
            ring_kernel=3,
        )
        rgb = torch.randn(2, 16, 8, 8, requires_grad=True)
        sar = torch.randn(2, 16, 8, 8, requires_grad=True)
        fused = torch.randn(2, 16, 8, 8, requires_grad=True)
        mask_a = torch.zeros(16, 16, dtype=torch.bool)
        mask_a[2:12, 2:12] = True
        mask_b = torch.zeros(8, 8, dtype=torch.bool)
        mask_b[1:7, 1:6] = True
        bags = [
            BuildingBagSpec(0, mask_a, label=1, area=100),
            BuildingBagSpec(1, mask_b, label=3, area=30),
        ]
        output = head(rgb, sar, fused, bags)
        self.assertEqual(output["exist_logits"].shape, (2, 2))
        self.assertEqual(output["severity_logits"].shape, (2, 2))
        self.assertEqual(output["labels"].tolist(), [1, 3])
        torch.testing.assert_close(output["attention_sum"], torch.ones(2), atol=1e-5, rtol=1e-5)
        self.assertGreater(output["intact_guard_logits"].numel(), 0)
        loss = -output["log_probs3"][torch.arange(2), torch.tensor([0, 2])].mean()
        loss.backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(sar.grad)
        self.assertGreater(float(rgb.grad.abs().sum()), 0.0)
        self.assertGreater(float(sar.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.local_exist.weight.grad)
        self.assertIsNotNone(head.local_severity.weight.grad)
        self.assertGreater(float(head.local_exist.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.local_severity.weight.grad.abs().sum()), 0.0)

    def test_sar_ring_changes_severity_evidence(self):
        torch.manual_seed(11)
        head = MICBDSMILHead(
            in_channels=16,
            token_dim=16,
            topk=2,
            max_tokens=32,
            ring_kernel=5,
            use_asymmetric_regions=True,
        ).eval()
        rgb = torch.zeros(1, 16, 9, 9)
        sar_a = torch.zeros_like(rgb)
        sar_b = sar_a.clone()
        fused = torch.zeros_like(rgb)
        mask = torch.zeros(9, 9, dtype=torch.bool)
        mask[3:6, 3:6] = True
        sar_b[:, :, 1:8, 1:8] = 2.0
        sar_b[:, :, 3:6, 3:6] = 0.0
        bag = [BuildingBagSpec(0, mask, label=2, area=9)]
        with torch.no_grad():
            severity_a = head(rgb, sar_a, fused, bag)["severity_logits"]
            severity_b = head(rgb, sar_b, fused, bag)["severity_logits"]
        self.assertFalse(torch.allclose(severity_a, severity_b))


if __name__ == "__main__":
    unittest.main()
