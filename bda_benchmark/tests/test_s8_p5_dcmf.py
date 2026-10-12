# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from model.DamageFormer_S8_P5_DCMF import (
    BidirectionalWindowCrossAttention,
    DecoupledCrossModalFusion,
    FrozenSARFilterBank,
    NativeSARHierarchicalEncoder,
    decur_redundancy_losses,
    get_p5_variant_spec,
)


class S8P5DCMFTests(unittest.TestCase):
    def test_filter_bank_is_fixed_finite_and_four_channel(self):
        module = FrozenSARFilterBank()
        value = module(torch.randn(2, 1, 32, 40))
        self.assertEqual(value.shape, (2, 4, 32, 40))
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_native_sar_encoder_has_independent_multiscale_contract(self):
        encoder = NativeSARHierarchicalEncoder(
            filter_augmentation=True, depths=(1, 1, 1, 1)
        )
        outputs = encoder(torch.randn(1, 1, 64, 64))
        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [
                (1, 96, 16, 16),
                (1, 192, 8, 8),
                (1, 384, 4, 4),
                (1, 768, 2, 2),
            ],
        )
        sum(value.mean() for value in outputs).backward()
        self.assertIsNotNone(encoder.stem[0].weight.grad)

    def test_window_cross_attention_handles_padding_and_gradient(self):
        module = BidirectionalWindowCrossAttention(
            channels=32, num_heads=4, window_size=8
        )
        rgb = torch.randn(2, 32, 15, 17, requires_grad=True)
        sar = torch.randn(2, 32, 15, 17, requires_grad=True)
        rgb_out, sar_out = module(rgb, sar)
        self.assertEqual(rgb_out.shape, rgb.shape)
        self.assertEqual(sar_out.shape, sar.shape)
        (rgb_out.mean() + sar_out.mean()).backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(sar.grad).all())

    def test_full_fusion_exposes_decur_features_and_finite_losses(self):
        module = DecoupledCrossModalFusion(
            cross_attention=True,
            common_unique_decoupling=True,
            common_dim=32,
            unique_dim=16,
            window_size=4,
        )
        rgb = torch.randn(2, 256, 12, 12, requires_grad=True)
        sar = torch.randn(2, 256, 12, 12, requires_grad=True)
        fused, auxiliary = module(rgb, sar)
        self.assertEqual(fused.shape, (2, 256, 12, 12))
        self.assertEqual(auxiliary["common_rgb"].shape[1], 32)
        self.assertEqual(auxiliary["unique_sar"].shape[1], 16)
        losses = decur_redundancy_losses({"fused_roi": fused, **auxiliary})
        total = fused.mean() + losses["loss_common"] + losses["loss_unique"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(sar.grad)

    def test_non_decur_variants_return_exact_zero_decur_loss(self):
        module = DecoupledCrossModalFusion(
            cross_attention=False,
            common_unique_decoupling=False,
        )
        fused, auxiliary = module(
            torch.randn(1, 256, 8, 8), torch.randn(1, 256, 8, 8)
        )
        losses = decur_redundancy_losses({"fused_roi": fused, **auxiliary})
        self.assertEqual(float(losses["loss_common"].item()), 0.0)
        self.assertEqual(float(losses["loss_unique"].item()), 0.0)

    def test_ablation_switches_are_nested_and_auditable(self):
        c0 = get_p5_variant_spec("c0")
        c1 = get_p5_variant_spec("c1")
        c2 = get_p5_variant_spec("c2")
        c3 = get_p5_variant_spec("c3")
        self.assertFalse(c0.filter_augmentation)
        self.assertTrue(c1.filter_augmentation)
        self.assertFalse(c1.cross_attention)
        self.assertTrue(c2.cross_attention)
        self.assertFalse(c2.common_unique_decoupling)
        self.assertTrue(c3.common_unique_decoupling)

    def test_decoupling_without_cross_attention_is_rejected(self):
        with self.assertRaises(ValueError):
            DecoupledCrossModalFusion(
                cross_attention=False,
                common_unique_decoupling=True,
            )


if __name__ == "__main__":
    unittest.main()
