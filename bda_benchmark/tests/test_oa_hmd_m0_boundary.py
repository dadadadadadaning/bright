import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    factorize_building_logits,
)
from bda_benchmark.script.standard_ML.m0_boundary_runtime import (
    apply_occurrence_offset,
    choose_global_candidate,
    global_boundary_gate,
    normalize_betas,
)


class OAHMDM0BoundaryTests(unittest.TestCase):
    def test_beta_sweep_requires_exact_zero_baseline(self):
        self.assertEqual(normalize_betas([0.1, 0.0, -0.1, 0.1]), (-0.1, 0.0, 0.1))
        with self.assertRaises(ValueError):
            normalize_betas([-0.2, 0.2])

    def test_zero_beta_is_exact_identity(self):
        damage = torch.randn(2, 4, 8, 8)
        location = torch.randn(2, 2, 8, 8)
        updated, _ = apply_occurrence_offset(damage, location, 0.0)
        self.assertTrue(torch.equal(updated, damage))

    def test_offset_preserves_damage_destroyed_odds(self):
        damage = torch.randn(2, 4, 8, 8)
        location = torch.zeros(2, 2, 8, 8)
        location[:, 1] = 1.0
        before_occurrence, before_severity = factorize_building_logits(damage)
        updated, _ = apply_occurrence_offset(damage, location, 0.35)
        after_occurrence, after_severity = factorize_building_logits(updated)
        self.assertTrue(torch.allclose(after_severity, before_severity, atol=1e-7))
        self.assertTrue(
            torch.allclose(
                after_occurrence,
                before_occurrence + 0.35,
                atol=1e-6,
            )
        )

    def test_offset_is_blocked_outside_frozen_p3_buildings(self):
        damage = torch.randn(1, 4, 4, 4)
        location = torch.zeros(1, 2, 4, 4)
        location[:, 0] = 1.0
        updated, support = apply_occurrence_offset(damage, location, 0.5)
        self.assertTrue(torch.equal(updated, damage))
        self.assertEqual(float(support.sum()), 0.0)

    def test_global_gate_requires_every_declared_condition(self):
        good = {
            "damaged_iou": 0.31,
            "affected_recall": 0.51,
            "damaged_gate_miss": -0.51,
            "valid_miou": 0.0,
            "event_mean": 0.0,
            "event_worst": 0.0,
        }
        self.assertTrue(global_boundary_gate(good, 10)["passed"])
        bad = dict(good, event_worst=-0.01)
        self.assertFalse(global_boundary_gate(bad, 10)["passed"])

    def test_candidate_prefers_a_passing_offset(self):
        def record(beta, passed, worst, event, damaged, valid):
            return {
                "beta": beta,
                "gate": {"passed": passed},
                "metrics": {
                    "event_worst_official_style_4class_miou": worst,
                    "event_mean_official_style_4class_miou": event,
                    "damaged_iou": damaged,
                    "valid_miou": valid,
                },
            }

        records = [
            record(0.0, False, 0.5, 0.5, 0.5, 0.5),
            record(0.1, False, 0.7, 0.7, 0.7, 0.7),
            record(0.2, True, 0.4, 0.4, 0.4, 0.4),
        ]
        self.assertEqual(float(choose_global_candidate(records)["beta"]), 0.2)


if __name__ == "__main__":
    unittest.main()
