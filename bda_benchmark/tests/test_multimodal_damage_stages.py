import unittest
from pathlib import Path

import torch
from torch import nn

from bda_benchmark.model.external.contracts import (
    ExternalModelContractError,
    normalize_external_output,
)
from bda_benchmark.model.external.flowmamba_bright_adapter import (
    FlowMambaBRIGHTAdapter,
)
from bda_benchmark.model.external.m2cd_bright_adapter import M2CDBRIGHTAdapter
from bda_benchmark.model.hybrid.p3_external_severity import (
    P3ExternalSeverityHybrid,
)
from bda_benchmark.model.losses.severity_distance_loss import (
    expected_severity_distance,
    masked_severity_cross_entropy,
)


class DummyP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, rgb, sar):
        loc = rgb.new_zeros((rgb.shape[0], 2, *rgb.shape[-2:]))
        loc[:, 1] = 2.0 * self.scale
        dam = rgb.new_zeros((rgb.shape[0], 4, *rgb.shape[-2:]))
        return loc, dam


class DummySeverity(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))

    def forward(self, rgb, sar):
        return self.bias.view(1, 3, 1, 1).expand(rgb.shape[0], 3, *rgb.shape[-2:])


class DummyExternal(nn.Module):
    def forward(self, first, second=None, **kwargs):
        source = first if second is None else second
        return source.new_zeros((source.shape[0], 4, *source.shape[-2:]))


class MultimodalDamageStageTests(unittest.TestCase):
    def test_dependency_audit_requires_real_sar_dino_checkpoint(self):
        source = Path(
            "bda_benchmark/script/standard_ML/"
            "audit_multimodal_damage_dependencies.py"
        ).read_text(encoding="utf-8")
        self.assertIn('record.get("checkpoint_required")', source)
        self.assertIn('record.get("checkpoint_ready")', source)

    def test_external_contract_derives_normalized_localization(self):
        logits = torch.randn(2, 4, 8, 8)
        output = normalize_external_output(logits, (8, 8))
        self.assertEqual(tuple(output["output_loc"].shape), (2, 2, 8, 8))
        p4 = logits.softmax(1)
        p2 = output["output_loc"].softmax(1)
        self.assertTrue(torch.allclose(p2[:, 1], p4[:, 1:].sum(1), atol=1e-6))

    def test_external_contract_rejects_binary_head(self):
        with self.assertRaises(ExternalModelContractError):
            normalize_external_output(torch.randn(1, 2, 4, 4), (4, 4))

    def test_r1_and_r2_adapters_expose_the_same_bright_contract(self):
        rgb = torch.randn(1, 3, 8, 8)
        sar = torch.rand(1, 2, 8, 8) * 255
        r1 = FlowMambaBRIGHTAdapter(DummyExternal())(rgb, sar, return_dict=True)
        r2 = M2CDBRIGHTAdapter(DummyExternal(), call_style="stacked")(
            rgb, sar, return_dict=True
        )
        for output in (r1, r2):
            self.assertEqual(tuple(output["output_dam"].shape), (1, 4, 8, 8))
            self.assertEqual(tuple(output["output_loc"].shape), (1, 2, 8, 8))

    def test_r3_freezes_p3_and_keeps_severity_trainable(self):
        model = P3ExternalSeverityHybrid(DummyP3(), DummySeverity())
        self.assertFalse(next(model.p3_locator.parameters()).requires_grad)
        self.assertTrue(next(model.severity_model.parameters()).requires_grad)
        model.train()
        self.assertFalse(model.p3_locator.training)

    def test_r3_probabilities_are_normalized_and_factorized(self):
        model = P3ExternalSeverityHybrid(DummyP3(), DummySeverity())
        rgb = torch.randn(2, 3, 8, 8)
        sar = torch.rand(2, 2, 8, 8) * 255
        output = model(rgb, sar, return_dict=True)
        probability = output["output_dam"].softmax(1)
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0])))
        building = output["output_loc"].softmax(1)[:, 1]
        self.assertTrue(torch.allclose(probability[:, 1:].sum(1), building, atol=1e-6))

    def test_distance_loss_respects_class_order(self):
        labels = torch.tensor([[[1, 3]]])
        correct = torch.tensor([[[[10.0, 0.0]], [[0.0, 0.0]], [[0.0, 10.0]]]])
        reversed_logits = correct.flip(1)
        self.assertLess(
            float(expected_severity_distance(correct, labels)),
            float(expected_severity_distance(reversed_logits, labels)),
        )

    def test_severity_losses_ignore_background(self):
        logits = torch.randn(1, 3, 2, 2, requires_grad=True)
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        loss = masked_severity_cross_entropy(logits, labels)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
