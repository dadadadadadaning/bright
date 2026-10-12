import unittest

import torch
import torch.nn as nn

from bda_benchmark.model.DamageFormer_RGBSAR_Q0_Readout import (
    Q0BuildingConditionedSARReadout,
    assert_q0_freeze_contract,
)
from bda_benchmark.script.standard_ML.q0_readout_runtime import (
    Q0MetricAccumulator,
    effective_number_weights,
    masked_three_class_ce,
    supervised_mask,
)


class _SAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, value):
        batch = value.shape[0]
        return [
            torch.zeros(batch, 96, 8, 8, device=value.device),
            torch.zeros(batch, 192, 4, 4, device=value.device),
            torch.zeros(batch, 384, 2, 2, device=value.device),
            torch.ones(batch, 768, 2, 2, device=value.device) * self.weight,
        ]


class _RGB(nn.Module):
    def forward(self, value):
        return [value, value, value, value]


class _Anchor(nn.Module):
    def forward(self, stages):
        rgb = stages[0]
        logits = torch.stack([-rgb[:, 0], rgb[:, 0]], dim=1)
        return rgb, logits


class _Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.variant = "a2"
        self.sar_encoder = _SAR()
        self.rgb_encoder = _RGB()
        self.p3_spatial_anchor = _Anchor()
        self.sar_aux_head = nn.Conv2d(768, 4, 1)


class Q0ReadoutTests(unittest.TestCase):
    def test_only_three_class_readout_is_trainable(self):
        model = Q0BuildingConditionedSARReadout(_Source(), "q0_a0")
        assert_q0_freeze_contract(model)
        self.assertEqual(model.trainable_parameter_names(), ["readout.weight", "readout.bias"])

    def test_source_stays_eval_when_wrapper_trains(self):
        model = Q0BuildingConditionedSARReadout(_Source(), "q0_a0")
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.source.training)

    def test_oracle_loss_ignores_background_exactly(self):
        damage = torch.tensor([[[0, 1], [2, 3]]])
        mask = supervised_mask(damage)
        logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
        first, count = masked_three_class_ce(logits, damage, mask)
        changed = logits.detach().clone()
        changed[:, :, 0, 0] = torch.tensor([1000.0, -1000.0, 500.0])
        second, second_count = masked_three_class_ce(changed, damage, mask)
        self.assertEqual(count, 3)
        self.assertEqual(second_count, 3)
        self.assertTrue(torch.equal(first.detach(), second.detach()))

    def test_p3_mask_only_changes_eligibility(self):
        damage = torch.tensor([[[1, 2], [3, 0]]])
        probability = torch.tensor([[[[0.9, 0.2], [0.7, 1.0]]]])
        mask = supervised_mask(damage, probability, threshold=0.5)
        self.assertTrue(torch.equal(mask, torch.tensor([[[True, False], [True, False]]])))

    def test_class_balancing_upweights_rare_cells(self):
        weights = effective_number_weights([10000, 100, 1000], beta=0.9999)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertLessEqual(float(weights.max() / weights.min()), 5.00001)

    def test_b0_reports_uncovered_damaged_as_end_to_end_miss(self):
        accumulator = Q0MetricAccumulator()
        damage = torch.tensor([[[2, 2]]])
        logits = torch.tensor([[[[0.0, 0.0]], [[5.0, 5.0]], [[0.0, 0.0]]]])
        mask = torch.tensor([[[True, False]]])
        accumulator.update(logits, damage, mask, ["demo-event_00000001"])
        result = accumulator.compute()
        self.assertEqual(result["conditional"]["damaged_recall"], 1.0)
        self.assertEqual(result["end_to_end_damaged_recall"], 0.5)
        self.assertEqual(result["building_coverage"], 0.5)

    def test_b0_forward_uses_frozen_rgb_mask_without_source_grad(self):
        model = Q0BuildingConditionedSARReadout(_Source(), "q0_b0")
        outputs = model(torch.ones(1, 3, 8, 8), torch.ones(1, 1, 8, 8))
        self.assertEqual(tuple(outputs["logits"].shape), (1, 3, 8, 8))
        self.assertFalse(outputs["stage4"].requires_grad)
        self.assertFalse(outputs["building_probability"].requires_grad)
        outputs["logits"].mean().backward()
        self.assertIsNone(model.source.sar_encoder.weight.grad)
        self.assertIsNotNone(model.readout.weight.grad)


if __name__ == "__main__":
    unittest.main()
