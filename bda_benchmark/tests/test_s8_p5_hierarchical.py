from __future__ import annotations

import unittest
import torch

from model.DamageFormer_S8_P5_Hierarchical import (
    HierarchicalChangeDecoder,
    apply_hierarchical_change,
    factorize_damage_logits,
)


class P5HierarchicalTests(unittest.TestCase):
    def test_factorization_has_expected_shape(self):
        occurrence, severity = factorize_damage_logits(torch.randn(2, 4, 8, 9))
        self.assertEqual(occurrence.shape, (2, 1, 8, 9))
        self.assertEqual(severity.shape, (2, 1, 8, 9))

    def test_zero_change_is_exact(self):
        base = torch.randn(2, 4, 7, 7)
        zero = torch.zeros_like(base[:, :1])
        self.assertTrue(torch.equal(apply_hierarchical_change(base, zero, zero), base))

    def test_exact_identity_keeps_transition_gradient(self):
        base = torch.randn(1, 4, 3, 3)
        occurrence = torch.zeros_like(base[:, :1], requires_grad=True)
        severity = torch.zeros_like(base[:, :1], requires_grad=True)
        output = apply_hierarchical_change(base, occurrence, severity)
        self.assertTrue(torch.equal(output, base))
        (output[:, 2].mean() + output[:, 3].mean()).backward()
        self.assertIsNotNone(occurrence.grad)
        self.assertIsNotNone(severity.grad)
        self.assertTrue(torch.isfinite(occurrence.grad).all())
        self.assertTrue(torch.isfinite(severity.grad).all())

    def test_occurrence_preserves_severity_odds(self):
        base = torch.randn(2, 4, 7, 7)
        occurrence = torch.randn_like(base[:, :1])
        zero = torch.zeros_like(occurrence)
        output = apply_hierarchical_change(base, occurrence, zero)
        self.assertTrue(torch.allclose(output[:, 3] - output[:, 2],
                                       base[:, 3] - base[:, 2], atol=1e-6))

    def test_severity_preserves_affected_mass(self):
        base = torch.randn(2, 4, 7, 7, dtype=torch.float64)
        zero = torch.zeros_like(base[:, :1])
        output = apply_hierarchical_change(base, zero, torch.randn_like(zero))
        old = torch.logsumexp(base[:, 2:4], 1)
        new = torch.logsumexp(output[:, 2:4], 1)
        self.assertTrue(torch.allclose(old, new, atol=1e-10))

    def test_decoder_starts_as_identity_and_receives_gradient(self):
        decoder = HierarchicalChangeDecoder(channels=16, hidden=8)
        base = torch.randn(2, 4, 8, 8, requires_grad=True)
        rgb = torch.randn(2, 16, 8, 8)
        sar = torch.randn(2, 16, 8, 8)
        fused = torch.randn(2, 16, 8, 8)
        output, aux = decoder(base, rgb, sar, fused, torch.ones(2, 1, 8, 8))
        self.assertTrue(torch.equal(output, base))
        output[:, 2].mean().backward()
        self.assertIsNotNone(decoder.occurrence_transition.grad)
        self.assertTrue(torch.isfinite(aux["occurrence_logits"]).all())


if __name__ == "__main__":
    unittest.main()
