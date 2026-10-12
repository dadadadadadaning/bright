import unittest
import tempfile
import json
from pathlib import Path

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H4 import DamageFormerRGBSARH4
from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    H6_VARIANTS,
    ClassConditionalSparseResidualRouter,
    DamageFormerRGBSARH6SDCR,
    SafeClassResidualRouter,
    SensorGapPrototypeAdapter,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (
    configure_freeze_policy,
    event_name,
    load_p3_rgb_anchor,
    tempered_event_balanced_sampler,
)


class H6SDCRTests(unittest.TestCase):
    def test_event_name_removes_only_the_chip_index(self):
        self.assertEqual(event_name("la_palma-volcano_00000123"), "la_palma-volcano")

    def test_tempered_event_sampler_uses_sqrt_inverse_frequency(self):
        class ToyDataset:
            data_list = ["large_1", "large_2", "large_3", "large_4", "small_1"]

        sampler = tempered_event_balanced_sampler(
            ToyDataset(), torch.Generator().manual_seed(1), power=0.5
        )
        weights = sampler.weights.tolist()
        self.assertEqual(weights[:4], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(weights[4], 1.0)

    def test_tempered_event_sampler_rejects_invalid_power(self):
        class ToyDataset:
            data_list = ["event_1"]

        with self.assertRaises(ValueError):
            tempered_event_balanced_sampler(
                ToyDataset(), torch.Generator().manual_seed(1), power=1.1
            )

    def test_h4_dual_sar_dataset_contract_is_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "normalization": {
                            "p01": 0.0,
                            "p99": 169.0,
                            "mean": 56.64234075291341,
                            "std": 45.34399136641941,
                        }
                    }
                ),
                encoding="utf-8",
            )
            dataset = HeterogeneousRGBSARDataset(
                str(root),
                ["contract_only"],
                64,
                split="val",
                variant="h4",
                sar_stats_path=str(stats),
            )
        self.assertEqual(dataset.variant, "h4")

    def test_variants_are_controlled_and_never_declare_teacher(self):
        self.assertEqual(
            tuple(H6_VARIANTS),
            ("a0", "a1", "a2", "a3", "b0", "b1", "b2", "b1r"),
        )
        self.assertFalse(H6_VARIANTS["a0"].p3_rgb_anchor)
        self.assertTrue(H6_VARIANTS["a1"].p3_rgb_anchor)
        self.assertTrue(H6_VARIANTS["a2"].sensor_gap_prototype)
        self.assertTrue(H6_VARIANTS["a3"].frequency_calibration)
        self.assertTrue(H6_VARIANTS["b0"].safe_p3_residual)
        self.assertEqual(H6_VARIANTS["b1"].sensor_stages, ("stage4",))
        self.assertEqual(
            H6_VARIANTS["b2"].sensor_stages, ("stage3", "stage4")
        )
        self.assertTrue(H6_VARIANTS["b1r"].class_sparse_router)
        self.assertEqual(H6_VARIANTS["b1r"].sensor_stages, ("stage4",))
        self.assertTrue(all(not value.sparse_fusion for value in H6_VARIANTS.values()))

    def test_a0_has_exact_h4_a4_state_contract(self):
        torch.manual_seed(4)
        h4 = DamageFormerRGBSARH4(variant="a4", pretrained_rgb=False)
        torch.manual_seed(4)
        h6 = DamageFormerRGBSARH6SDCR(variant="a0", pretrained_rgb=False)
        self.assertEqual(set(h4.state_dict()), set(h6.state_dict()))
        h6.load_state_dict(h4.state_dict(), strict=True)

    def test_constant_sensor_gap_is_removed_by_prototype(self):
        adapter = SensorGapPrototypeAdapter(8, frequency_calibration=False)
        joint = torch.randn(2, 8, 7, 7)
        rgb = torch.randn(2, 8, 7, 7)
        sar = rgb + 3.25
        unique_rgb = torch.randn_like(rgb)
        unique_sar = torch.randn_like(rgb)
        building = torch.zeros(2, 1, 7, 7)
        intact = torch.ones(2, 1, 7, 7)
        output, diagnostics = adapter(
            joint, rgb, sar, unique_rgb, unique_sar, building, intact
        )
        self.assertTrue(torch.equal(output, joint))
        self.assertLess(float(diagnostics["sensor_gap_corrected_abs"].max()), 1e-6)
        self.assertTrue(torch.isfinite(diagnostics["sensor_gap_prototype"]).all())

    def test_frequency_branch_is_exactly_safe_at_initialization(self):
        adapter = SensorGapPrototypeAdapter(8, frequency_calibration=True)
        tensors = [torch.randn(1, 8, 8, 8) for _ in range(5)]
        building = torch.rand(1, 1, 8, 8)
        intact = torch.rand(1, 1, 8, 8)
        output, diagnostics = adapter(*tensors, building, intact)
        self.assertTrue(torch.equal(output, tensors[0]))
        self.assertEqual(float(diagnostics["causal_transition"].item()), 0.0)
        self.assertEqual(float(diagnostics["frequency_transition"].item()), 0.0)
        self.assertTrue(torch.isfinite(diagnostics["frequency_gate"]).all())

    def test_zero_transition_is_safe_but_not_gradient_dead(self):
        adapter = SensorGapPrototypeAdapter(8, frequency_calibration=False)
        values = [torch.randn(1, 8, 8, 8) for _ in range(5)]
        output, _ = adapter(
            *values,
            torch.rand(1, 1, 8, 8),
            torch.rand(1, 1, 8, 8),
        )
        output.sum().backward()
        self.assertIsNotNone(adapter.causal_transition.grad)
        self.assertGreater(float(adapter.causal_transition.grad.abs()), 0.0)

    def test_safe_class_router_is_exact_and_not_gradient_dead(self):
        router = SafeClassResidualRouter()
        p3 = torch.randn(2, 4, 16, 16)
        native = torch.randn_like(p3)
        output, diagnostics = router(p3, native)
        self.assertTrue(torch.equal(output, p3))
        self.assertEqual(
            tuple(diagnostics["safe_class_transition"].shape),
            (1, 4, 1, 1),
        )
        output[:, 1:].sum().backward()
        self.assertIsNotNone(router.class_transition.grad)
        self.assertGreater(float(router.class_transition.grad.abs().sum()), 0.0)

    def test_native_sar_inverse_reconstructs_p3_pseudo_rgb(self):
        model = DamageFormerRGBSARH6SDCR(variant="b0", pretrained_rgb=False)
        display = torch.tensor([[[[0.0, 56.0], [169.0, 255.0]]]])
        native = (
            display.clamp_max(169.0) - model.native_sar_mean
        ) / model.native_sar_std
        audit = display
        reconstructed = model._p3_pseudo_rgb_sar(
            torch.cat([native, audit], dim=1)
        )
        expected = (
            display.repeat(1, 3, 1, 1) - model.imagenet_mean
        ) / model.imagenet_std
        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-6))

    def test_sparse_router_is_exact_low_open_and_class_conditional(self):
        router = ClassConditionalSparseResidualRouter()
        p3 = torch.randn(2, 4, 16, 16)
        native = torch.randn_like(p3)
        output, diagnostics = router(p3, native)
        gates = diagnostics["safe_residual_sar_gate"]
        transition = diagnostics["safe_class_transition"]
        self.assertTrue(torch.equal(output, p3))
        self.assertEqual(tuple(gates.shape), (2, 4, 16, 16))
        self.assertTrue(torch.equal(gates[:, 0], torch.zeros_like(gates[:, 0])))
        self.assertLess(float(gates[:, 1:].max()), 0.019)
        self.assertLessEqual(float(transition.abs().max()), 0.10)
        output[:, 1:].sum().backward()
        self.assertGreater(float(router.class_transition.grad.abs().sum()), 0.0)

    def test_b1r_freezes_everything_except_sparse_router(self):
        model = DamageFormerRGBSARH6SDCR(variant="b1r", pretrained_rgb=False)
        configure_freeze_policy(model)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("safe_residual_router") for name in trainable)
        )

    def test_p3_rgb_checkpoint_key_migration_is_strict(self):
        model = DamageFormerRGBSARH6SDCR(variant="a1", pretrained_rgb=False)
        state = {}
        for key, value in model.rgb_encoder.features.state_dict().items():
            state[f"encoder_1.swin_transformer.features.{key}"] = value.clone()
        for key, value in model.p3_spatial_anchor.fusion.state_dict().items():
            state[f"fusion_layer_1.{key}"] = value.clone()
        for key, value in model.p3_spatial_anchor.head.state_dict().items():
            state[f"clf_1.{key}"] = value.clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p3.pth"
            torch.save(
                {"model": state, "experiment": "S8_MICB_DSMIL_P3", "step": 4000},
                path,
            )
            record = load_p3_rgb_anchor(model, str(path))
        self.assertGreater(record["loaded_encoder_keys"], 0)
        self.assertGreater(record["loaded_fusion_keys"], 0)
        self.assertFalse(record["teacher"])

    def test_rgb_localization_is_independent_of_sar(self):
        torch.manual_seed(9)
        model = DamageFormerRGBSARH6SDCR(variant="a3", pretrained_rgb=False).eval()
        rgb = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            first = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
            second = model(rgb, torch.randn(1, 1, 64, 64), return_dict=True)
        self.assertTrue(torch.equal(first["output_loc"], second["output_loc"]))

    def test_freeze_policy_keeps_p3_anchor_and_bn_frozen(self):
        model = DamageFormerRGBSARH6SDCR(variant="a3", pretrained_rgb=False)
        configure_freeze_policy(model)
        for name, parameter in model.named_parameters():
            if name.startswith(
                ("rgb_encoder", "p3_spatial_anchor", "p3_full_anchor")
            ):
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(
            any(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith("sensor_damage_adapters")
            )
        )
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_sar_freeze_ablation_changes_only_declared_depth(self):
        expectations = {
            "f0_shallow_frozen": {
                "sar_encoder.stem.0.0.weight": False,
                "sar_encoder.stages.0.0.depthwise.weight": False,
                "sar_encoder.stages.1.0.depthwise.weight": False,
                "sar_encoder.stages.2.0.depthwise.weight": True,
            },
            "f1_stage2_to_4": {
                "sar_encoder.stem.0.0.weight": False,
                "sar_encoder.stages.0.0.depthwise.weight": False,
                "sar_encoder.stages.1.0.depthwise.weight": True,
                "sar_encoder.stages.2.0.depthwise.weight": True,
            },
            "f2_all_sar": {
                "sar_encoder.stem.0.0.weight": True,
                "sar_encoder.stages.0.0.depthwise.weight": True,
                "sar_encoder.stages.1.0.depthwise.weight": True,
                "sar_encoder.stages.2.0.depthwise.weight": True,
            },
        }
        for mode, expected in expectations.items():
            model = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
            configure_freeze_policy(model, mode=mode)
            parameters = dict(model.named_parameters())
            for name, trainable in expected.items():
                self.assertEqual(parameters[name].requires_grad, trainable, (mode, name))
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for name, parameter in parameters.items()
                    if name.startswith("rgb_encoder")
                )
            )


if __name__ == "__main__":
    unittest.main()
