import unittest

import torch
import torch.nn as nn

from bda_benchmark.model.DamageFormer_RGBSAR_H11_SATR import (
    H11_VARIANTS,
    SARMAEEncoder,
    SARMAEResidualPyramid,
)
from bda_benchmark.script.standard_ML.train_H11_SATR import (
    configure_h11_freeze_policy,
    migrate_sarmae_encoder_state,
)


class _TinyFreezeContract(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.rgb_encoder = nn.Linear(2, 2)
        self.hierarchy_transition = nn.Parameter(torch.tensor(0.0))
        self.sar_encoder = nn.Module()
        self.sar_encoder.transfer_adapters = nn.Linear(2, 2)
        self.sar_encoder.transfer_transition = nn.ParameterDict(
            {"stage3": nn.Parameter(torch.tensor(0.0))}
        )
        self.sar_encoder.sarmae_encoder = nn.Module()
        self.sar_encoder.sarmae_encoder.blocks = nn.ModuleList(
            [nn.Linear(2, 2) for _ in range(12)]
        )
        self.sar_encoder.sarmae_encoder.norm = nn.LayerNorm(2)


class H11SATRTests(unittest.TestCase):
    def test_variants_form_a_single_controlled_transfer_chain(self):
        self.assertFalse(H11_VARIANTS["t0"].transfer)
        self.assertTrue(H11_VARIANTS["t1"].transfer)
        self.assertEqual(H11_VARIANTS["t1"].trainable_sarmae_blocks, 0)
        self.assertEqual(H11_VARIANTS["t2"].trainable_sarmae_blocks, 2)
        self.assertFalse(H11_VARIANTS["t1"].exact_official_input)
        self.assertTrue(H11_VARIANTS["t1e"].exact_official_input)
        self.assertEqual(H11_VARIANTS["t1e"].trainable_sarmae_blocks, 0)
        self.assertEqual(H11_VARIANTS["t2e"].trainable_sarmae_blocks, 2)

    def test_mae_encoder_accepts_native_single_channel_and_is_spatial(self):
        encoder = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        first, second = encoder(torch.randn(2, 1, 40, 56))
        self.assertEqual(tuple(first.shape), (2, 32, 5, 7))
        self.assertEqual(tuple(second.shape), (2, 32, 5, 7))
        self.assertTrue(torch.isfinite(second).all())

    def test_rgb_style_patch_weights_are_safely_collapsed_to_one_channel(self):
        source = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=3,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        destination = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=1,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        checkpoint = {"model": source.state_dict()}
        expected = source.patch_embed.proj.weight.detach().mean(1, keepdim=True)
        report = migrate_sarmae_encoder_state(destination, checkpoint)
        self.assertTrue(report["strict_encoder_coverage"])
        self.assertTrue(torch.allclose(destination.patch_embed.proj.weight, expected))

    def test_official_joint_checkpoint_sar_encoder_prefix_is_supported(self):
        source = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=3,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        destination = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=1,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        checkpoint = {
            "model": {
                **{
                    f"sar_encoder.{key}": value
                    for key, value in source.state_dict().items()
                },
                "optical_encoder.cls_token": torch.randn(1, 1, 32),
                "sar_alignment_ffn.0.weight": torch.randn(32, 32),
            }
        }
        report = migrate_sarmae_encoder_state(destination, checkpoint)
        self.assertTrue(report["strict_encoder_coverage"])
        self.assertEqual(report["encoder_blocks"], 2)
        self.assertTrue(
            torch.allclose(
                destination.patch_embed.proj.weight,
                source.patch_embed.proj.weight.mean(1, keepdim=True),
            )
        )

    def test_exact_input_migration_preserves_official_three_channel_patch_embed(self):
        source = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=3,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        destination = SARMAEEncoder(
            image_size=32,
            patch_size=8,
            in_channels=3,
            channels=32,
            depth=2,
            heads=4,
            tap_blocks=(0, 1),
        )
        checkpoint = {
            "model": {
                f"sar_encoder.{key}": value
                for key, value in source.state_dict().items()
            }
        }
        report = migrate_sarmae_encoder_state(destination, checkpoint)
        self.assertEqual(report["patch_input_channels"], 3)
        self.assertTrue(
            torch.equal(
                destination.patch_embed.proj.weight,
                source.patch_embed.proj.weight,
            )
        )

    def test_exact_official_photometric_input_matches_public_transform(self):
        pyramid = SARMAEResidualPyramid(
            exact_official_input=True,
            encoder_kwargs={
                "image_size": 32,
                "patch_size": 8,
                "channels": 32,
                "depth": 2,
                "heads": 4,
                "tap_blocks": (0, 1),
            },
        )
        display = torch.tensor([[[[0.0, 255.0]]]])
        observed = pyramid.official_sarmae_input(display)
        unit = display.repeat(1, 3, 1, 1) / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(observed, (unit - mean) / std))

    def test_exact_input_zero_transition_preserves_native_pyramid(self):
        pyramid = SARMAEResidualPyramid(
            exact_official_input=True,
            encoder_kwargs={
                "image_size": 32,
                "patch_size": 8,
                "channels": 32,
                "depth": 2,
                "heads": 4,
                "tap_blocks": (0, 1),
            },
        )
        native_sar = torch.randn(1, 1, 32, 32)
        display_sar = torch.rand(1, 1, 32, 32) * 255.0
        expected = super(SARMAEResidualPyramid, pyramid).forward(native_sar)
        pyramid.set_display_sar_context(display_sar)
        observed = pyramid(native_sar)
        for actual, reference in zip(observed, expected):
            self.assertTrue(torch.equal(actual, reference))

    def test_exact_input_fails_closed_without_raw_display_channel(self):
        pyramid = SARMAEResidualPyramid(
            exact_official_input=True,
            encoder_kwargs={
                "image_size": 32,
                "patch_size": 8,
                "channels": 32,
                "depth": 2,
                "heads": 4,
                "tap_blocks": (0, 1),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "two-channel dataset contract"):
            pyramid(torch.randn(1, 1, 32, 32))

    def test_zero_transfer_is_exact_but_transition_receives_gradient(self):
        pyramid = SARMAEResidualPyramid(
            encoder_kwargs={
                "image_size": 32,
                "patch_size": 8,
                "channels": 32,
                "depth": 2,
                "heads": 4,
                "tap_blocks": (0, 1),
            }
        )
        sar = torch.randn(1, 1, 32, 32)
        native = super(SARMAEResidualPyramid, pyramid).forward(sar)
        output = pyramid(sar)
        for observed, expected in zip(output, native):
            self.assertTrue(torch.equal(observed, expected))
        output[-1].mean().backward()
        gradient = pyramid.transfer_transition["stage4"].grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs()), 0.0)

    def test_t1_trains_only_adapters_and_transitions(self):
        model = _TinyFreezeContract("t1")
        configure_h11_freeze_policy(model)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                "transfer_adapters" in name or "transfer_transition" in name
                for name in trainable
            )
        )

    def test_t2_unfreezes_only_the_last_two_mae_blocks(self):
        model = _TinyFreezeContract("t2")
        configure_h11_freeze_policy(model)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(any("blocks.10" in name for name in trainable))
        self.assertTrue(any("blocks.11" in name for name in trainable))
        self.assertFalse(any("blocks.9" in name for name in trainable))
        self.assertFalse(any(name.startswith("rgb_encoder") for name in trainable))

    def test_t2e_uses_the_same_restricted_tail_finetuning_policy(self):
        model = _TinyFreezeContract("t2e")
        configure_h11_freeze_policy(model)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertTrue(any("blocks.10" in name for name in trainable))
        self.assertTrue(any("blocks.11" in name for name in trainable))
        self.assertFalse(any("blocks.9" in name for name in trainable))


if __name__ == "__main__":
    unittest.main()
