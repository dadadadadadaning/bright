import unittest

import torch

from changedetection.experiments.p3_transfer.model import (
    apply_soft_building_roi,
    factorized_damage_log_probs,
)


class ChangeMambaP3TransferTests(unittest.TestCase):
    def test_zero_roi_transition_is_exact_identity(self):
        feature = torch.randn(2, 8, 9, 7, requires_grad=True)
        building = torch.rand(2, 1, 9, 7)
        transition = torch.zeros((), requires_grad=True)
        observed, weight = apply_soft_building_roi(
            feature, building, transition, roi_floor=0.1
        )
        self.assertTrue(torch.equal(observed, feature))
        self.assertTrue(torch.equal(weight, torch.ones_like(weight)))
        observed.square().mean().backward()
        self.assertIsNotNone(transition.grad)
        self.assertTrue(torch.isfinite(transition.grad))

    def test_zero_hierarchy_residual_preserves_four_class_probabilities(self):
        logits = torch.randn(3, 4, 11, 13)
        zero = torch.zeros_like(logits[:, :1])
        observed = factorized_damage_log_probs(logits, zero, zero).softmax(1)
        expected = logits.softmax(1)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-6, rtol=2e-6))

    def test_occurrence_changes_no_background_or_severity_odds(self):
        logits = torch.randn(2, 4, 5, 6)
        occurrence = torch.full_like(logits[:, :1], 0.7)
        zero = torch.zeros_like(occurrence)
        base = logits.softmax(1)
        changed = factorized_damage_log_probs(logits, occurrence, zero).softmax(1)
        self.assertTrue(torch.allclose(base[:, :1], changed[:, :1], atol=2e-6))
        base_odds = base[:, 3:4] / base[:, 2:3].clamp_min(1e-8)
        changed_odds = changed[:, 3:4] / changed[:, 2:3].clamp_min(1e-8)
        self.assertTrue(torch.allclose(base_odds, changed_odds, atol=2e-5, rtol=2e-5))

    def test_severity_changes_no_background_or_affected_mass(self):
        logits = torch.randn(2, 4, 5, 6)
        severity = torch.full_like(logits[:, :1], -0.4)
        zero = torch.zeros_like(severity)
        base = logits.softmax(1)
        changed = factorized_damage_log_probs(logits, zero, severity).softmax(1)
        self.assertTrue(torch.allclose(base[:, :1], changed[:, :1], atol=2e-6))
        self.assertTrue(
            torch.allclose(
                base[:, 2:4].sum(1), changed[:, 2:4].sum(1), atol=2e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
