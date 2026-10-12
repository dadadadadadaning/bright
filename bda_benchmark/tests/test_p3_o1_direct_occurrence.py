# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from script.standard_ML.p3_direct_occurrence_runtime import (
    affected_logit,
    balanced_occurrence_bce,
    o1_validation_checks,
)
from script.standard_ML.train_P3_O1_direct_occurrence import (
    load_o1_model_delta,
    o1_model_delta,
)


class P3O1DirectOccurrenceTests(unittest.TestCase):
    class _ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.fusion_layer_3 = nn.Linear(2, 2)
            self.clf_2 = nn.Linear(2, 2)

    def test_affected_logit_is_exact_log_probability_odds(self):
        logits = torch.tensor([[[[9.0]], [[1.0]], [[2.0]], [[3.0]]]])
        expected = torch.logsumexp(torch.tensor([2.0, 3.0]), dim=0) - 1.0
        self.assertTrue(torch.allclose(affected_logit(logits)[0, 0, 0], expected))

    def test_balanced_loss_ignores_background_and_ignore(self):
        logits = torch.zeros(1, 4, 1, 4, requires_grad=True)
        target = torch.tensor([[[0, 1, 2, 255]]])
        loss, diagnostics = balanced_occurrence_bce(logits, target)
        loss.backward()
        self.assertGreater(float(logits.grad[:, :, :, 1:3].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, :, 3].abs().sum()), 0.0)
        self.assertEqual(diagnostics["intact_pixels"], 1)
        self.assertEqual(diagnostics["affected_pixels"], 1)

    def test_correct_occurrence_margins_reduce_loss(self):
        target = torch.tensor([[[1, 2, 3]]])
        bad = torch.zeros(1, 4, 1, 3)
        good = bad.clone()
        good[:, 1, 0, 0] = 6.0
        good[:, 2, 0, 1] = 6.0
        good[:, 3, 0, 2] = 6.0
        bad_loss, _ = balanced_occurrence_bce(bad, target)
        good_loss, _ = balanced_occurrence_bce(good, target)
        self.assertLess(float(good_loss), float(bad_loss))

    def test_delta_checkpoint_contains_only_inference_layers(self):
        source = self._ToyModel()
        delta = o1_model_delta(source)
        self.assertTrue(delta)
        self.assertTrue(
            all(
                name.startswith(("fusion_layer_3.", "clf_2."))
                for name in delta
            )
        )
        target = self._ToyModel()
        encoder_before = target.encoder.weight.detach().clone()
        load_o1_model_delta(target, delta)
        self.assertTrue(torch.equal(target.encoder.weight, encoder_before))
        self.assertTrue(
            torch.equal(
                target.fusion_layer_3.weight,
                source.fusion_layer_3.weight,
            )
        )

    def test_gate_requires_direct_occurrence_improvement(self):
        baseline = {
            "valid_miou": 0.60,
            "damaged_iou": 0.40,
            "intact_iou": 0.75,
            "destroyed_iou": 0.60,
            "damaged_to_intact": 0.30,
            "damaged_gate_miss": 0.39,
            "affected_recall": 0.70,
            "event_worst_official_style_4class_miou": 0.40,
            "event_miou_by_name": {str(index): 0.50 for index in range(14)},
        }
        candidate = dict(baseline)
        candidate.update(
            {
                "valid_miou": 0.601,
                "damaged_iou": 0.404,
                "damaged_to_intact": 0.29,
                "damaged_gate_miss": 0.38,
                "affected_recall": 0.71,
                "event_worst_official_style_4class_miou": 0.401,
                "event_miou_by_name": {
                    str(index): 0.501 for index in range(14)
                },
            }
        )
        checks, _ = o1_validation_checks(candidate, baseline)
        self.assertTrue(all(checks.values()))
        candidate["damaged_to_intact"] = 0.31
        checks, _ = o1_validation_checks(candidate, baseline)
        self.assertFalse(checks["damaged_to_intact_decreasing"])


if __name__ == "__main__":
    unittest.main()

