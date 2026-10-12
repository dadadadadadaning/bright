# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from model.cmr.occurrence import CMRTrainingOnlyOccurrenceProjector
from script.standard_ML.p3_hard_occurrence_runtime import (
    differentiable_region_mean_key,
    h1_validation_checks,
    match_cached_component,
    occurrence_info_nce,
    protected_spatial_kl,
)
from script.standard_ML.train_CMR_H1_occurrence import (
    h1_model_delta,
    load_h1_model_delta,
)


class CMRH1OccurrenceTests(unittest.TestCase):
    class _ToyH1Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.fusion_layer_3 = nn.Linear(2, 2)
            self.clf_2 = nn.Linear(2, 2)

    def test_delta_checkpoint_contains_only_h1_inference_layers(self):
        source = self._ToyH1Model()
        delta = h1_model_delta(source)
        self.assertTrue(delta)
        self.assertTrue(
            all(
                name.startswith(("fusion_layer_3.", "clf_2."))
                for name in delta
            )
        )
        target = self._ToyH1Model()
        frozen_before = target.encoder.weight.detach().clone()
        load_h1_model_delta(target, delta)
        self.assertTrue(torch.equal(target.encoder.weight, frozen_before))
        self.assertTrue(
            torch.equal(target.fusion_layer_3.weight, source.fusion_layer_3.weight)
        )

    def test_projector_is_exact_identity_at_initialisation(self):
        torch.manual_seed(3)
        model = CMRTrainingOnlyOccurrenceProjector(8, 4)
        feature = torch.randn(5, 8)
        expected = torch.nn.functional.normalize(feature, dim=1)
        self.assertTrue(torch.allclose(model(feature), expected, atol=1e-7))

    def test_region_mean_key_is_differentiable(self):
        feature = torch.arange(16.0).reshape(1, 1, 4, 4).requires_grad_(True)
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[:2, :2] = True
        key = differentiable_region_mean_key(feature, mask, batch_index=0)
        key.sum().backward()
        self.assertIsNotNone(feature.grad)
        self.assertTrue(bool(torch.isfinite(feature.grad).all()))

    def test_component_match_requires_exact_bbox_and_area(self):
        mask = np.zeros((6, 6), dtype=np.bool_)
        mask[1:3, 2:5] = True
        component = match_cached_component(mask, [2, 1, 5, 3], 6)
        self.assertIsNotNone(component)
        self.assertIsNone(match_cached_component(mask, [2, 1, 5, 4], 6))

    def test_occurrence_loss_backpropagates(self):
        query = torch.randn(3, 8, requires_grad=True)
        positive = torch.randn(3, 8)
        negative = torch.randn(3, 4, 8)
        loss = occurrence_info_nce(
            query, positive, negative, temperature=0.1
        )
        loss.backward()
        self.assertGreater(float(query.grad.abs().sum()), 0.0)

    def test_protected_kl_ignores_damaged_pixels(self):
        teacher = torch.zeros(1, 4, 1, 2)
        student = teacher.clone()
        student[:, 2, 0, 0] = 10.0  # pixel 0 is Damaged and must be ignored
        target = torch.tensor([[[2, 1]]])
        loss = protected_spatial_kl(student, teacher, target)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_validation_gate_requires_all_declared_conditions(self):
        baseline = {
            "valid_miou": 0.60,
            "damaged_iou": 0.40,
            "intact_iou": 0.75,
            "destroyed_iou": 0.60,
            "damaged_gate_miss": 0.30,
            "affected_recall": 0.70,
            "event_worst_official_style_4class_miou": 0.40,
            "event_miou_by_name": {str(i): 0.5 for i in range(14)},
        }
        candidate = dict(baseline)
        candidate.update({
            "valid_miou": 0.601,
            "damaged_iou": 0.404,
            "damaged_gate_miss": 0.29,
            "affected_recall": 0.71,
            "event_worst_official_style_4class_miou": 0.401,
            "event_miou_by_name": {str(i): 0.501 for i in range(14)},
        })
        checks, _ = h1_validation_checks(candidate, baseline)
        self.assertTrue(all(checks.values()))
        candidate["intact_iou"] = 0.74
        checks, _ = h1_validation_checks(candidate, baseline)
        self.assertFalse(checks["intact_iou_drop_within_limit"])

    def test_validation_gate_requires_all_14_events(self):
        baseline = {
            "valid_miou": 0.60,
            "damaged_iou": 0.40,
            "intact_iou": 0.75,
            "destroyed_iou": 0.60,
            "damaged_gate_miss": 0.30,
            "affected_recall": 0.70,
            "event_worst_official_style_4class_miou": 0.40,
            "event_miou_by_name": {str(i): 0.5 for i in range(13)},
        }
        candidate = dict(baseline)
        candidate.update({
            "damaged_iou": 0.404,
            "damaged_gate_miss": 0.29,
        })
        checks, _ = h1_validation_checks(candidate, baseline)
        self.assertFalse(checks["all_14_events_present"])


if __name__ == "__main__":
    unittest.main()
