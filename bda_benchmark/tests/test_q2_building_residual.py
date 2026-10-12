import unittest

import torch
import torch.nn as nn

from bda_benchmark.model.DamageFormer_RGBSAR_Q2_BuildingResidual import (
    Q2BuildingResidualFusion,
    assert_q2_freeze_contract,
    build_p3_instance_map,
)
from bda_benchmark.script.standard_ML.q2_building_residual_runtime import (
    advancement_gate,
    region_cross_entropy,
    region_targets,
)


class DummyP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, rgb, sar):
        batch, _, height, width = rgb.shape
        loc = rgb.new_full((batch, 2, height, width), -4.0)
        loc[:, 0] = 4.0
        loc[:, 1, 2:6, 2:6] = 8.0
        loc[:, 0, 2:6, 2:6] = -8.0
        damage = rgb.new_zeros((batch, 4, height, width))
        damage[:, 1] = 2.0
        return loc, damage


class DummySAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, sar):
        stage = sar.repeat(1, 4, 1, 1) * self.scale
        return [stage, stage, stage, stage]


class DummySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.sar_encoder = DummySAREncoder()


class DummyQ0(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = DummySource()
        self.readout = nn.Conv2d(4, 3, 1)


class Q2BuildingResidualTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        model = Q2BuildingResidualFusion(
            DummyP3(), DummyQ0(), "q2_b1", minimum_area=2
        )
        return model

    def inputs(self):
        return torch.randn(1, 3, 8, 8), torch.randn(1, 2, 8, 8)

    def test_safe_initialization_is_bitwise_p3(self):
        model = self.make_model()
        rgb, sar = self.inputs()
        outputs = model(rgb, sar, return_dict=True)
        self.assertTrue(torch.equal(outputs["output_dam"], outputs["p3_output_dam"]))
        self.assertEqual(float(outputs["q2_transition"]), 0.0)

    def test_only_region_head_and_transition_train(self):
        model = self.make_model()
        assert_q2_freeze_contract(model)
        names = model.trainable_parameter_names()
        self.assertIn("transition_raw", names)
        self.assertTrue(all(name == "transition_raw" or name.startswith("region_head.") for name in names))

    def test_zero_transition_has_first_step_gradient(self):
        model = self.make_model()
        with torch.no_grad():
            model.region_head[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
        rgb, sar = self.inputs()
        damage = torch.full((1, 8, 8), 255, dtype=torch.long)
        damage[:, 2:6, 2:6] = 2
        outputs = model(rgb, sar, return_dict=True)
        loss, count = region_cross_entropy(
            outputs, damage,
            minimum_building_fraction=1.0, minimum_class_purity=1.0,
        )
        self.assertEqual(count, 1)
        loss.backward()
        self.assertIsNotNone(model.transition_raw.grad)
        self.assertGreater(abs(float(model.transition_raw.grad)), 0.0)

    def test_residual_is_blocked_outside_p3_building(self):
        model = self.make_model()
        with torch.no_grad():
            model.transition_raw.fill_(1.0)
            model.region_head[-1].weight.zero_()
            model.region_head[-1].bias.copy_(torch.tensor([0.0, 2.0, 0.0]))
        rgb, sar = self.inputs()
        outputs = model(rgb, sar, return_dict=True)
        instances = outputs["q2_instance_map"]
        outside = instances == 0
        self.assertTrue(torch.equal(
            outputs["output_dam"].permute(0, 2, 3, 1)[outside],
            outputs["p3_output_dam"].permute(0, 2, 3, 1)[outside],
        ))
        self.assertFalse(torch.equal(
            outputs["output_dam"][:, :, 3, 3],
            outputs["p3_output_dam"][:, :, 3, 3],
        ))

    def test_region_target_is_four_class_building_label(self):
        logits = torch.full((1, 2, 4, 4), -4.0)
        logits[:, 0] = 4.0
        logits[:, 1, 1:3, 1:3] = 8.0
        logits[:, 0, 1:3, 1:3] = -8.0
        instance_map = build_p3_instance_map(logits, minimum_area=1)
        records = [type("Record", (), {"batch_index": 0, "instance_id": 1})()]
        damage = torch.full((1, 4, 4), 255, dtype=torch.long)
        damage[:, 1:3, 1:3] = 2
        indices, labels = region_targets(
            records, instance_map, damage,
            minimum_building_fraction=1.0, minimum_class_purity=1.0,
        )
        self.assertEqual(indices.tolist(), [0])
        self.assertEqual(labels.tolist(), [2])

    def test_advancement_requires_all_safety_checks(self):
        base = {
            "valid_miou": 0.60, "miou4": 0.69, "damaged_iou": 0.40,
            "event_mean_official_style_4class_miou": 0.50,
            "event_worst_official_style_4class_miou": 0.40,
            "damaged_to_intact": 0.30,
        }
        final = dict(base)
        final.update({
            "valid_miou": 0.604, "damaged_iou": 0.406,
            "event_mean_official_style_4class_miou": 0.501,
            "event_worst_official_style_4class_miou": 0.401,
            "damaged_to_intact": 0.29,
        })
        audit = {"nonnegative_events": 11, "region_net_correction": 3}
        self.assertTrue(advancement_gate(final, base, audit)["passed"])
        audit["region_net_correction"] = -1
        self.assertFalse(advancement_gate(final, base, audit)["passed"])


if __name__ == "__main__":
    unittest.main()
