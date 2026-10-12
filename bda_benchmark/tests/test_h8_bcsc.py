import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
    SensorGapPrototypeAdapter,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H8_BCSC import (
    BuildingConditionedSensorAdapter,
    DamageFormerRGBSARH8BCSC,
    H8_VARIANTS,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (
    configure_freeze_policy,
)


class H8BCSCTests(unittest.TestCase):
    @staticmethod
    def inputs(channels=8, size=8):
        joint = torch.randn(2, channels, size, size)
        rgb = torch.randn_like(joint)
        sar = torch.randn_like(joint)
        rgb_unique = torch.randn_like(joint)
        sar_unique = torch.randn_like(joint)
        building = torch.rand(2, 1, size, size)
        intact = torch.rand(2, 1, size, size)
        return joint, rgb, sar, rgb_unique, sar_unique, building, intact

    def test_variants_are_controlled(self):
        self.assertEqual(tuple(H8_VARIANTS), ("c0", "c1", "c2"))
        self.assertFalse(H8_VARIANTS["c0"].building_conditioned_calibration)
        self.assertTrue(H8_VARIANTS["c1"].building_conditioned_calibration)
        self.assertFalse(H8_VARIANTS["c1"].reliability_gate)
        self.assertTrue(H8_VARIANTS["c2"].reliability_gate)

    def test_c0_has_exact_h6_a2_state_contract(self):
        torch.manual_seed(17)
        a2 = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
        torch.manual_seed(17)
        c0 = DamageFormerRGBSARH8BCSC(variant="c0", pretrained_rgb=False)
        self.assertEqual(set(a2.state_dict()), set(c0.state_dict()))
        c0.load_state_dict(a2.state_dict(), strict=True)

    def test_c1_migration_adds_only_prototype_blends(self):
        source = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
        target = DamageFormerRGBSARH8BCSC(variant="c1", pretrained_rgb=False)
        incompatible = target.load_state_dict(source.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all("prototype_blend" in key for key in incompatible.missing_keys)
        )

    def test_c1_adapter_is_exact_a2_at_initialization(self):
        base = SensorGapPrototypeAdapter(8, frequency_calibration=False)
        refined = BuildingConditionedSensorAdapter(8, reliability_gate=False)
        incompatible = refined.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(incompatible.missing_keys, ["prototype_blend"])
        self.assertFalse(incompatible.unexpected_keys)
        with torch.no_grad():
            base.causal_transition.fill_(0.45)
            refined.causal_transition.fill_(0.45)
        values = self.inputs()
        base_output, _ = base(*values)
        refined_output, diagnostics = refined(*values)
        self.assertTrue(torch.equal(base_output, refined_output))
        self.assertEqual(float(diagnostics["bcsc_prototype_blend"]), 0.0)

    def test_prototype_refinement_has_a_first_step_gradient(self):
        adapter = BuildingConditionedSensorAdapter(8, reliability_gate=False)
        with torch.no_grad():
            adapter.causal_transition.fill_(0.5)
        output, _ = adapter(*self.inputs())
        output.square().mean().backward()
        self.assertIsNotNone(adapter.prototype_blend.grad)
        self.assertTrue(torch.isfinite(adapter.prototype_blend.grad))
        self.assertGreater(float(adapter.prototype_blend.grad.abs()), 0.0)

    def test_c2_gate_is_bounded_and_building_conditioned(self):
        adapter = BuildingConditionedSensorAdapter(8, reliability_gate=True)
        values = list(self.inputs())
        values[-2] = torch.zeros_like(values[-2])
        _, diagnostics = adapter(*values)
        gate = diagnostics["bcsc_reliability_gate"]
        self.assertTrue(torch.equal(gate, torch.zeros_like(gate)))

        values[-2] = torch.ones_like(values[-2])
        _, diagnostics = adapter(*values)
        gate = diagnostics["bcsc_reliability_gate"]
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), adapter.maximum_gate)
        self.assertLess(float(gate.mean()), 0.01)

    def test_empty_building_support_is_finite(self):
        adapter = BuildingConditionedSensorAdapter(8, reliability_gate=False)
        values = list(self.inputs())
        values[-2] = torch.zeros_like(values[-2])
        output, diagnostics = adapter(*values)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(
            torch.isfinite(diagnostics["bcsc_building_prototype"]).all()
        )
        self.assertEqual(
            float(diagnostics["bcsc_prototype_confidence"].max()), 0.0
        )

    def test_oracle_context_is_transient_and_evaluation_only(self):
        model = DamageFormerRGBSARH8BCSC(
            variant="c1", pretrained_rgb=False
        )
        rgb = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 1, 64, 64)
        oracle = torch.ones(1, 1, 64, 64)
        with self.assertRaises(RuntimeError):
            model(
                rgb,
                sar,
                return_dict=True,
                audit_oracle_intact_mask=oracle,
                audit_forced_blend=0.5,
            )
        model.eval()
        with torch.no_grad():
            model(
                rgb,
                sar,
                return_dict=True,
                audit_oracle_intact_mask=oracle,
                audit_forced_blend=0.5,
            )
        for adapter in model.sensor_damage_adapters.values():
            self.assertIsNone(adapter._audit_oracle_intact_mask)
            self.assertIsNone(adapter._audit_forced_blend)

    def test_freeze_policy_keeps_rgb_and_localization_frozen(self):
        model = DamageFormerRGBSARH8BCSC(variant="c1", pretrained_rgb=False)
        configure_freeze_policy(model)
        for name, parameter in model.named_parameters():
            if name.startswith(("rgb_encoder", "p3_spatial_anchor")):
                self.assertFalse(parameter.requires_grad, name)
        blends = [
            parameter
            for name, parameter in model.named_parameters()
            if "prototype_blend" in name
        ]
        self.assertTrue(blends)
        self.assertTrue(all(parameter.requires_grad for parameter in blends))


if __name__ == "__main__":
    unittest.main()
