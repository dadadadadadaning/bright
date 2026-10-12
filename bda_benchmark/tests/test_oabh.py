import tempfile
import unittest
from pathlib import Path

import torch

from bda_benchmark.dataset.event_balanced_sampling import (
    EventBalancedBatchSampler,
    event_balanced_weights,
    event_from_sample_name,
    expected_event_mass,
)

from bda_benchmark.model.DamageFormer_RGBSAR_OABH import (
    CrossEventSARStyleMix,
    EventRobustNativeSARPyramid,
    OpticalAnchoredBuildingHierarchy,
    SAREventStyleSeparation,
    SoftDamageEvidenceConditioner,
    WindowCrossAttention2d,
    load_h6_encoder_initialisation,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import (
    checkpoint_selection_key,
)


class OABHTests(unittest.TestCase):
    def test_cross_event_mixstyle_is_train_only_and_sar_only(self):
        module = CrossEventSARStyleMix(probability=1.0, alpha=0.1)
        feature = torch.cat(
            [torch.randn(1, 8, 9, 9), 5.0 + torch.randn(1, 8, 9, 9)], dim=0
        )
        module.eval()
        self.assertTrue(torch.equal(module(feature, ("a", "b")), feature))
        module.train()
        mixed = module(feature, ("a", "b"))
        self.assertEqual(mixed.shape, feature.shape)
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertFalse(torch.equal(mixed, feature))

    def test_style_separation_is_exact_identity_with_live_transition(self):
        module = SAREventStyleSeparation(16)
        feature = torch.randn(2, 16, 8, 8, requires_grad=True)
        output = module(feature)
        self.assertTrue(torch.equal(output, feature))
        output.square().mean().backward()
        self.assertIsNotNone(module.transition.grad)
        self.assertGreater(float(module.transition.grad.abs()), 0.0)

    def test_style_block_is_inserted_before_deeper_sar_stages(self):
        baseline = EventRobustNativeSARPyramid("none")
        adapted = EventRobustNativeSARPyramid("adapter")
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("style_blocks.") for key in missing))
        self.assertFalse(unexpected)
        sar = torch.randn(1, 1, 64, 64)
        baseline.eval()
        adapted.eval()
        with torch.no_grad():
            before = baseline(sar)
            safe = adapted(sar)
            adapted.style_blocks[0].transition.fill_(0.5)
            opened = adapted(sar)
        for left, right in zip(before, safe):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(torch.equal(opened[-1], safe[-1]))

    def test_event_name_parser_preserves_multiword_disaster_name(self):
        self.assertEqual(
            event_from_sample_name("la_palma-volcano_00000123"),
            "la_palma-volcano",
        )

    def test_event_balancing_equalises_expected_event_mass(self):
        names = [
            "event-a_00000001",
            "event-a_00000002",
            "event-a_00000003",
            "event-b_00000001",
        ]
        weights, counts = event_balanced_weights(names)
        self.assertEqual(counts, {"event-a": 3, "event-b": 1})
        self.assertEqual(weights, [1 / 3, 1 / 3, 1 / 3, 1.0])
        mass = expected_event_mass(names)
        self.assertAlmostEqual(mass["event-a"], 0.5)
        self.assertAlmostEqual(mass["event-b"], 0.5)

    def test_event_balanced_batches_use_distinct_events(self):
        names = [
            "event-a_00000001", "event-a_00000002",
            "event-b_00000001", "event-c_00000001", "event-d_00000001",
        ]
        sampler = EventBalancedBatchSampler(names, batch_size=3, num_batches=8)
        for batch in sampler:
            events = {event_from_sample_name(names[index]) for index in batch}
            self.assertEqual(len(events), 3)

    def test_window_cross_attention_is_finite_and_differentiable(self):
        module = WindowCrossAttention2d(32, heads=4, window_size=5)
        rgb = torch.randn(2, 32, 11, 13, requires_grad=True)
        sar = torch.randn(2, 32, 11, 13, requires_grad=True)
        output = module(rgb, sar)
        self.assertEqual(output.shape, rgb.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(sar.grad)
        self.assertGreater(float(sar.grad.abs().sum()), 0.0)

    def test_evidence_is_soft_building_conditioned_and_multiscale(self):
        module = SoftDamageEvidenceConditioner(32)
        destruction = torch.randn(1, 32, 8, 8)
        occurrence = torch.randn(1, 32, 16, 16, requires_grad=True)
        support = torch.zeros(1, 1, 8, 8)
        context, evidence, coverage = module(destruction, occurrence, support)
        self.assertEqual(context.shape, destruction.shape)
        self.assertEqual(evidence.shape, (1, 1, 16, 16))
        self.assertEqual(coverage.shape, (1, 3, 16, 16))
        self.assertTrue(torch.equal(coverage, torch.zeros_like(coverage)))
        context.mean().backward()
        self.assertGreater(float(occurrence.grad.abs().sum()), 0.0)

    def test_variants_form_declared_architecture_chain(self):
        n0 = OpticalAnchoredBuildingHierarchy("n0", pretrained_rgb=False)
        self.assertEqual(n0.occurrence_path.stage_indices, (1, 2))
        self.assertEqual(n0.destruction_path.stage_indices, (2, 3))
        self.assertIsNone(n0.shared_attention)
        del n0
        shared = OpticalAnchoredBuildingHierarchy("n1_shared", pretrained_rgb=False)
        self.assertIsNotNone(shared.shared_attention)
        self.assertIsNone(shared.occurrence_attention)
        del shared
        stage = OpticalAnchoredBuildingHierarchy("n1_stage", pretrained_rgb=False)
        self.assertIsNotNone(stage.occurrence_attention)
        self.assertIsNotNone(stage.destruction_attention)
        self.assertIsNot(stage.occurrence_attention, stage.destruction_attention)
        self.assertIsNone(stage.severity_conditioner)
        del stage
        conditioned = OpticalAnchoredBuildingHierarchy("n1_cond", pretrained_rgb=False)
        self.assertIsNotNone(conditioned.severity_conditioner)
        self.assertIsNone(conditioned.evidence_conditioner)
        del conditioned
        n2 = OpticalAnchoredBuildingHierarchy("n2", pretrained_rgb=False)
        self.assertIsNotNone(n2.evidence_conditioner)

    def test_model_has_no_p3_router_or_logit_residual(self):
        model = OpticalAnchoredBuildingHierarchy("n2", pretrained_rgb=False)
        names = "\n".join(name.lower() for name, _ in model.named_modules())
        for forbidden in ("p3", "router", "transition", "residual", "flow", "conv3d"):
            self.assertNotIn(forbidden, names)

    def test_occurrence_and_destruction_do_not_share_heads_or_relations(self):
        model = OpticalAnchoredBuildingHierarchy("n2", pretrained_rgb=False)
        self.assertIsNot(model.occurrence_path, model.destruction_path)
        self.assertIsNot(model.occurrence_relation, model.destruction_relation)
        self.assertIsNot(model.occurrence_head, model.severity_head)

    def test_factorised_probabilities_are_normalised(self):
        model = OpticalAnchoredBuildingHierarchy("n0", pretrained_rgb=False)
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 1, 64, 64),
                return_dict=True,
            )
        probabilities = outputs["output_dam"].softmax(1)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(
            torch.allclose(
                probabilities.sum(1),
                torch.ones_like(probabilities[:, 0]),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertEqual(outputs["output_loc"].shape, (1, 2, 64, 64))
        self.assertEqual(outputs["output_dam"].shape, (1, 4, 64, 64))

    def test_phase_contract_changes_encoder_trainability_only(self):
        model = OpticalAnchoredBuildingHierarchy("n1_stage", pretrained_rgb=False)
        model.set_training_phase("heads")
        self.assertFalse(any(p.requires_grad for p in model.rgb_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.sar_encoder.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.occurrence_head.parameters()))
        model.set_training_phase("deep")
        self.assertFalse(any(p.requires_grad for p in model.rgb_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.localization_anchor.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.sar_encoder.stages[-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.sar_encoder.stem.parameters()))

        adapted = OpticalAnchoredBuildingHierarchy(
            "n1_shared", pretrained_rgb=False, sar_style_mode="adapter"
        )
        adapted.set_training_phase("deep")
        self.assertTrue(
            any(p.requires_grad for p in adapted.sar_encoder.style_blocks.parameters())
        )

    def test_h6_initialisation_loads_only_encoder_namespaces(self):
        source = OpticalAnchoredBuildingHierarchy("n0", pretrained_rgb=False)
        target = OpticalAnchoredBuildingHierarchy("n1_stage", pretrained_rgb=False)
        state = {
            "model": {
                **{f"rgb_encoder.{k}": v for k, v in source.rgb_encoder.state_dict().items()},
                **{f"sar_encoder.{k}": v for k, v in source.sar_encoder.state_dict().items()},
                **{
                    f"p3_spatial_anchor.{k}": v
                    for k, v in source.localization_anchor.state_dict().items()
                },
                "damage_head.unused": torch.ones(1),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake_h6.pth"
            torch.save(state, path)
            report = load_h6_encoder_initialisation(target, path)
        self.assertGreater(report["rgb_encoder_loaded"], 0)
        self.assertGreater(report["sar_encoder_loaded"], 0)
        self.assertGreater(report["localization_anchor_loaded"], 0)

    def test_event_worst_selector_uses_declared_tie_breakers(self):
        first = {
            "event_worst_official_style_4class_miou": 0.40,
            "event_mean_official_style_4class_miou": 0.50,
            "damaged_iou": 0.45,
            "valid_miou": 0.60,
        }
        second = dict(first)
        second["event_mean_official_style_4class_miou"] = 0.51
        self.assertGreater(
            checkpoint_selection_key(
                second, "event_worst_official_style_4class_miou"
            ),
            checkpoint_selection_key(
                first, "event_worst_official_style_4class_miou"
            ),
        )


if __name__ == "__main__":
    unittest.main()
