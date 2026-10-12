# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.make_data_loader_p4_native_sar import (
    VALID_VARIANTS as P4_DATASET_VARIANTS,
    make_sar_views,
    normalize_native_sar,
    to_single_channel_sar,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    NativeSARConvStem,
    P4_VARIANTS,
    collapse_pseudo_rgb_patch_weight,
    configure_p4_trainable,
    fold_p3_imagenet_patch_embed_to_native,
    get_p4_variant_spec,
)
from script.standard_ML.compute_S8_P4_sar_stats import UniformReservoir


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        holder = nn.Module()
        holder.features = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(1, 4, 1), nn.GroupNorm(1, 4)),
                nn.Sequential(nn.Conv2d(4, 4, 1)),
                nn.Sequential(nn.Conv2d(4, 4, 1)),
                nn.Sequential(nn.Conv2d(4, 4, 1)),
                nn.Sequential(nn.Conv2d(4, 4, 1)),
            ]
        )
        self.swin_transformer = holder


class _FakeP4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_1 = nn.Conv2d(3, 4, 1)
        self.encoder_2 = _Encoder()
        self.fusion_layer_1 = nn.Conv2d(4, 4, 1)
        self.fusion_layer_2 = nn.Conv2d(4, 4, 1)
        self.fusion_layer_3 = nn.Sequential(
            nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4)
        )
        self.clf_1 = nn.Conv2d(4, 2, 1)
        self.clf_2 = nn.Conv2d(4, 4, 1)
        self.micb_dsmil = nn.Conv2d(4, 4, 1)


class S8P4NativeSARTests(unittest.TestCase):
    def test_model_and_dataset_share_one_variant_registry(self):
        self.assertEqual(tuple(P4_VARIANTS), tuple(P4_DATASET_VARIANTS))

    def test_a0_is_exact_p3_teacher_view(self):
        sar = np.arange(30, dtype=np.float32).reshape(5, 6)
        student, teacher = make_sar_views(sar, "a0", None)
        np.testing.assert_array_equal(student, teacher)
        self.assertEqual(student.shape, (3, 5, 6))

    def test_train_stats_views_have_declared_channels(self):
        sar = np.arange(20, dtype=np.float32).reshape(4, 5)
        stats = {"p01": 2.0, "p99": 17.0, "mean": 9.5, "std": 4.0}
        normalized = normalize_native_sar(sar, stats)
        a1, _ = make_sar_views(sar, "a1", stats)
        a2, _ = make_sar_views(sar, "a2", stats)
        a3, _ = make_sar_views(sar, "a3", stats)
        a4, _ = make_sar_views(sar, "a4", stats)
        self.assertEqual(a1.shape, (3, 4, 5))
        self.assertEqual(a2.shape, (1, 4, 5))
        self.assertEqual(a3.shape, (1, 4, 5))
        self.assertEqual(a4.shape, (1, 4, 5))
        np.testing.assert_allclose(a1[0], normalized)
        np.testing.assert_allclose(a1[0], a1[2])
        np.testing.assert_allclose(a2[0], normalized)
        # A4 is deliberately affine and unclipped so its patch embed can be
        # exactly folded from P3's pseudo-RGB contract.
        expected_a4 = (sar - stats["mean"]) / stats["std"]
        np.testing.assert_allclose(a4[0], expected_a4)
        self.assertNotEqual(float(a2[0, 0, 0]), float(a4[0, 0, 0]))

    def test_nonidentical_rgb_sar_is_rejected(self):
        image = np.zeros((3, 4, 3), dtype=np.float32)
        image[..., 1] = 1.0
        with self.assertRaises(ValueError):
            to_single_channel_sar(image)

    def test_a2_weight_collapse_is_exact_for_repeated_input(self):
        torch.manual_seed(3)
        scalar = torch.randn(2, 1, 20, 20)
        repeated = scalar.repeat(1, 3, 1, 1)
        weight = torch.randn(8, 3, 4, 4)
        bias = torch.randn(8)
        expected = F.conv2d(repeated, weight, bias, stride=4)
        actual = F.conv2d(
            scalar, collapse_pseudo_rgb_patch_weight(weight), bias, stride=4
        )
        torch.testing.assert_close(actual, expected)

    def test_a4_affine_fold_matches_p3_imagenet_patch_embed(self):
        torch.manual_seed(7)
        dtype = torch.float64
        raw = torch.randn(2, 1, 20, 24, dtype=dtype) * 35.0 + 110.0
        weight = torch.randn(8, 3, 4, 4, dtype=dtype)
        bias = torch.randn(8, dtype=dtype)
        imagenet_mean = torch.tensor(
            [123.675, 116.28, 103.53], dtype=dtype
        ).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(
            [58.395, 57.12, 57.375], dtype=dtype
        ).view(1, 3, 1, 1)
        pseudo = (raw.repeat(1, 3, 1, 1) - imagenet_mean) / imagenet_std
        expected = F.conv2d(pseudo, weight, bias, stride=4)

        sar_mean, sar_std = 91.25, 22.5
        native_input = (raw - sar_mean) / sar_std
        native_weight, native_bias = fold_p3_imagenet_patch_embed_to_native(
            weight,
            bias,
            sar_mean=sar_mean,
            sar_std=sar_std,
        )
        actual = F.conv2d(
            native_input, native_weight, native_bias, stride=4
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_a3_stem_shape_and_finite_output(self):
        stem = NativeSARConvStem()
        # The inherited P3 feature extractor records only nn.Sequential
        # blocks. Keep the A3 replacement inside that interface contract so
        # encoder_2 continues returning patch_embed + four Swin stages.
        self.assertIsInstance(stem, nn.Sequential)
        output = stem(torch.randn(2, 1, 64, 80))
        self.assertEqual(output.shape, (2, 16, 20, 96))
        self.assertTrue(torch.isfinite(output).all())

    def test_freeze_policy_is_exact_and_bn_stays_frozen(self):
        model = _FakeP4()
        configure_p4_trainable(model, input_only=True)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("encoder_2.swin_transformer.features.0.")
                for name in trainable
            )
        )

        configure_p4_trainable(model, input_only=False)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertFalse(
            any(
                name.startswith("encoder_2.swin_transformer.features.1.")
                for name in trainable
            )
        )
        self.assertTrue(any(name.startswith("fusion_layer_3.0.") for name in trainable))
        self.assertTrue(any(name.startswith("clf_2.") for name in trainable))
        self.assertFalse(any(name.startswith("encoder_1.") for name in trainable))
        self.assertFalse(any(name.startswith("clf_1.") for name in trainable))
        self.assertFalse(any(name.startswith("micb_dsmil.") for name in trainable))
        self.assertFalse(model.fusion_layer_3[1].training)
        self.assertFalse(model.fusion_layer_3[1].weight.requires_grad)

    def test_variants_do_not_declare_reliability_mechanisms(self):
        for variant, channels in (
            ("a0", 3), ("a1", 3), ("a2", 1), ("a3", 1), ("a4", 1)
        ):
            spec = get_p4_variant_spec(variant)
            self.assertEqual(spec.sar_channels, channels)
            text = repr(spec).lower()
            self.assertNotIn("reliability", text)
            self.assertNotIn("boundary", text)

    def test_reservoir_supports_more_than_one_billion_seen_pixels(self):
        reservoir = UniformReservoir(capacity=64, seed=3407)
        reservoir.update(np.arange(128, dtype=np.float32))
        # Exercise the NumPy-limit fallback without allocating one billion
        # values. The reservoir invariant is already established above.
        reservoir.seen = 1_000_000_001
        reservoir.update(np.arange(256, dtype=np.float32))
        self.assertEqual(reservoir.sample().shape, (64,))
        self.assertEqual(reservoir.large_population_fallback_draws, 1)
        self.assertTrue(np.isfinite(reservoir.sample()).all())

    def test_joint_no_mil_control_is_protocol_locked(self):
        runner = Path(__file__).parents[1] / "run" / "run_S8_P4_NSMICB_5090.sh"
        text = runner.read_text(encoding="utf-8")
        block = text.split("  joint_no_mil)", 1)[1].split("    ;;", 1)[0]
        self.assertIn("EXPECTED_RESUME_STEP=1000", block)
        self.assertIn("EXPECTED_RESUME_VALID_MIOU_PERCENT", block)
        self.assertIn("ADMIT_RESUME_CHECKPOINT=1", block)
        self.assertIn("LAMBDA_MIL=0.0", block)
        self.assertIn("LAMBDA_EXIST_AUX=0.0", block)
        self.assertIn("LAMBDA_SEVERITY_AUX=0.0", block)
        self.assertIn("train_run 1500 250 1000", block)


if __name__ == "__main__":
    unittest.main()
