# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.DamageFormer_P3_S2Occurrence import (
    DamageFormerP3Stage2Occurrence,
    Stage2LocalOccurrenceEncoder,
    occurrence_log_odds,
    occurrence_only_update,
)
from script.standard_ML.p3_s2_occurrence_runtime import (
    hard_stage2_occurrence_loss,
)


class P3Stage2OccurrenceTests(unittest.TestCase):
    def test_zero_delta_is_exact_p3_guard(self):
        base = torch.randn(2, 4, 7, 9)
        output = occurrence_only_update(base, torch.zeros_like(base[:, :1]))
        self.assertTrue(torch.equal(output, base))

    def test_update_preserves_background_building_mass_and_severity(self):
        base = torch.randn(2, 4, 5, 6)
        delta = torch.randn(2, 1, 5, 6) * 0.2
        output = occurrence_only_update(base, delta)
        self.assertTrue(torch.equal(output[:, 0], base[:, 0]))
        self.assertTrue(
            torch.allclose(
                torch.logsumexp(output[:, 1:4], dim=1),
                torch.logsumexp(base[:, 1:4], dim=1),
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output[:, 3] - output[:, 2],
                base[:, 3] - base[:, 2],
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                occurrence_log_odds(output),
                occurrence_log_odds(base) + delta,
                atol=2e-6,
                rtol=2e-6,
            )
        )

    def test_stage2_encoder_is_zero_initialised_and_building_gated(self):
        encoder = Stage2LocalOccurrenceEncoder()
        rgb = torch.randn(2, 192, 8, 8)
        sar = torch.randn(2, 192, 8, 8)
        support = torch.ones(2, 1, 16, 16)
        delta, aux = encoder(rgb, sar, support)
        self.assertEqual(tuple(delta.shape), (2, 1, 8, 8))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertEqual(tuple(aux["stage2_local_tokens"].shape), (2, 64, 8, 8))

        with torch.no_grad():
            encoder.occurrence_head.bias.fill_(1.0)
        zero_support = torch.zeros_like(support)
        gated, _ = encoder(rgb, sar, zero_support)
        self.assertTrue(torch.equal(gated, torch.zeros_like(gated)))

    def test_hard_loss_skips_batch_without_both_classes(self):
        occurrence = torch.zeros(1, 1, 3, 3)
        target = torch.full((1, 3, 3), 2)
        loss, diagnostics = hard_stage2_occurrence_loss(
            occurrence,
            occurrence,
            occurrence,
            target,
            minimum_hard_pixels=1,
        )
        self.assertIsNone(loss)
        self.assertFalse(diagnostics["usable"])

    def test_hard_loss_updates_selected_damaged_and_intact(self):
        new = torch.zeros(1, 1, 1, 6, requires_grad=True)
        base = torch.tensor([[[[-2.0, -1.0, 0.0, 0.2, 0.5, 1.0]]]])
        delta = new
        target = torch.tensor([[[2, 2, 2, 1, 1, 3]]])
        loss, diagnostics = hard_stage2_occurrence_loss(
            new,
            base,
            delta,
            target,
            hard_fraction=0.5,
            minimum_hard_pixels=1,
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertGreater(float(new.grad.abs().sum()), 0.0)
        self.assertTrue(diagnostics["usable"])
        self.assertEqual(diagnostics["selected_per_class"], 2)

    def test_wrapper_freezes_p3_and_trains_only_occurrence_encoder(self):
        model = DamageFormerP3Stage2Occurrence()
        self.assertFalse(any(parameter.requires_grad for parameter in model.p3.parameters()))
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.occurrence_encoder.parameters())
        )
        model.train()
        self.assertFalse(model.p3.training)
        self.assertTrue(model.occurrence_encoder.training)


if __name__ == "__main__":
    unittest.main()
