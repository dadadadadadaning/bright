import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    VALID_HETEROGENEOUS_RGBSAR_VARIANTS,
)
from bda_benchmark.model.DamageFormer_P3_SRAC_Controlled import (
    ClassBalancedRegionMemory,
    LearnableLocalSuperpixels,
    P3SRACControlled,
    VALID_P3_SRAC_VARIANTS,
)
from bda_benchmark.script.standard_ML.p3_srac_controlled_runtime import (
    node_class_weights,
    structure_region_supervision,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import (
    checkpoint_selection_key,
)
from bda_benchmark.script.standard_ML.summarize_P3_SRAC import (
    effective_structure_transition,
)
from bda_benchmark.script.standard_ML.summarize_P3_SRAC_partition import (
    build_report as build_partition_report,
)
from bda_benchmark.script.standard_ML.summarize_P3_SRAC_affinity import (
    build_report as build_affinity_report,
)


class TinyP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Conv2d(3, 256, 1)
        self.sar = nn.Conv2d(3, 256, 1)
        self.loc = nn.Conv2d(256, 2, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)

    def forward(self, pre, post, return_aux=False):
        rgb = F.avg_pool2d(self.rgb(pre), 4)
        sar = F.avg_pool2d(self.sar(post), 4)
        loc_low = self.loc(rgb)
        p_building = loc_low.softmax(1)[:, 1:2]
        fused = 0.5 * (rgb + sar)
        fused_roi = fused * (0.1 + 0.9 * p_building)
        damage_low = self.clf_2(fused_roi)
        output_loc = F.interpolate(loc_low, pre.shape[-2:], mode="bilinear")
        output_dam = F.interpolate(damage_low, pre.shape[-2:], mode="bilinear")
        if not return_aux:
            return output_loc, output_dam
        return output_loc, output_dam, {
            "rgb_feature": rgb,
            "sar_feature": sar,
            "fused_roi": fused_roi,
            "p_building": p_building,
        }


class P3SRACControlledTests(unittest.TestCase):
    def test_model_and_dataset_share_exact_controlled_variant_registry(self):
        self.assertEqual(
            tuple(VALID_P3_SRAC_VARIANTS),
            tuple(
                value
                for value in VALID_HETEROGENEOUS_RGBSAR_VARIANTS
                if value.startswith("sr")
            ),
        )

    def setUp(self):
        torch.manual_seed(7)
        self.rgb = torch.randn(2, 3, 32, 32)
        # Channel one is the historical raw display SAR.
        self.sar = torch.stack(
            (torch.randn(2, 32, 32), torch.rand(2, 32, 32) * 255.0), dim=1
        )

    def test_dataset_contract_accepts_every_controlled_variant(self):
        required = {
            "sr0", "sr1", "sr1f", "sr1p", "sr1pa", "sr1pb",
            "sr1pai", "sr1pib", "sr1paa", "sr2",
        }
        available = set(VALID_HETEROGENEOUS_RGBSAR_VARIANTS)
        missing = sorted(required - available)
        self.assertFalse(
            missing,
            "dataset variant registry is stale; missing: "
            f"{missing}; available: {sorted(available)}",
        )

    def test_sr0_is_exact_p3(self):
        p3 = TinyP3()
        model = P3SRACControlled(p3, "sr0", relation_channels=16)
        expected = p3(self.rgb, self.sar[:, -1:].repeat(1, 3, 1, 1))
        # P3SRAC reconstructs the historical normalisation; use its own output
        # as the exact reference to test that SR0 does not alter the tensors.
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )

        expected = p3(self.rgb, P3PseudoSARSource.to_p3_input(self.sar))
        observed = model(self.rgb, self.sar)
        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertTrue(torch.equal(expected[1], observed[1]))

    def test_learned_assignment_is_local_normalised_and_trainable(self):
        module = LearnableLocalSuperpixels(16, 4, cell_size=2)
        feature = torch.randn(2, 16, 8, 8, requires_grad=True)
        assignment = module(feature)
        self.assertTrue(
            torch.allclose(
                assignment.weights.sum(1),
                torch.ones_like(assignment.weights[:, 0]),
                atol=1e-6,
            )
        )
        assignment.entropy.mean().backward()
        self.assertIsNotNone(module.embedding[0].weight.grad)
        self.assertTrue(torch.isfinite(module.embedding[0].weight.grad).all())

    def test_iterative_assignment_updates_seeds_and_keeps_gradient(self):
        single = LearnableLocalSuperpixels(
            16, 4, cell_size=2, boundary_aware=True,
            assignment_iterations=1,
        )
        iterative = LearnableLocalSuperpixels(
            16, 4, cell_size=2, boundary_aware=True,
            assignment_iterations=3,
        )
        iterative.load_state_dict(single.state_dict(), strict=True)
        feature = torch.randn(2, 16, 8, 8, requires_grad=True)
        assignment_single = single(feature)
        assignment_iterative = iterative(feature)
        self.assertTrue(
            torch.allclose(
                assignment_iterative.weights.sum(1),
                torch.ones_like(assignment_iterative.weights[:, 0]),
                atol=1.0e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                assignment_single.weights,
                assignment_iterative.weights,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        )
        self.assertIn("srac_q0_q1_l1", iterative.last_diagnostics)
        self.assertIn("srac_q1_q2_l1", iterative.last_diagnostics)
        self.assertIn("srac_assignment_top12_margin", iterative.last_diagnostics)
        self.assertGreaterEqual(
            float(iterative.last_diagnostics["srac_q0_q1_l1"]), 0.0
        )
        assignment_iterative.entropy.mean().backward()
        gradient = iterative.embedding[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_sr1_probabilities_are_factorised_and_region_head_gets_gradient(self):
        model = P3SRACControlled(TinyP3(), "sr1", cell_size=2, relation_channels=16)
        outputs = model(self.rgb, self.sar, return_dict=True)
        probability = outputs["output_dam"].exp()
        self.assertTrue(
            torch.allclose(
                probability.sum(1), torch.ones_like(probability[:, 0]), atol=1e-5
            )
        )
        labels = torch.randint(0, 4, (2, 32, 32))
        losses = structure_region_supervision(outputs, labels, 255, model=model)
        losses["representation_loss"].backward()
        gradient = model.region_classifier.classifier[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_sr1f_preserves_p3_background_probability_exactly(self):
        p3 = TinyP3()
        model = P3SRACControlled(p3, "sr1f", cell_size=2, relation_channels=16)
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )

        _loc, base_damage = p3(
            self.rgb, P3PseudoSARSource.to_p3_input(self.sar)
        )
        outputs = model(self.rgb, self.sar, return_dict=True)
        expected_background = base_damage.softmax(1)[:, 0]
        observed_background = outputs["output_dam"].exp()[:, 0]
        self.assertTrue(
            torch.allclose(expected_background, observed_background, atol=1.0e-6)
        )

    def test_sr1p_safe_state_matches_p3_probabilities_and_transition_gets_gradient(self):
        p3 = TinyP3()
        model = P3SRACControlled(p3, "sr1p", cell_size=2, relation_channels=16)
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )

        _loc, base_damage = p3(
            self.rgb, P3PseudoSARSource.to_p3_input(self.sar)
        )
        outputs = model(self.rgb, self.sar, return_dict=True)
        expected = base_damage.softmax(1)
        observed = outputs["output_dam"].exp()
        self.assertTrue(torch.allclose(expected, observed, atol=1.0e-6))
        self.assertEqual(float(outputs["srac_structure_transition"]), 0.0)

        labels = torch.randint(0, 4, (2, 32, 32))
        loss = F.cross_entropy(outputs["output_dam"], labels)
        loss.backward()
        gradient = model.structure_transition.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs()), 0.0)
        diagnostics = structure_region_supervision(
            outputs, labels, 255, model=model
        )
        self.assertEqual(float(diagnostics["representation_loss"]), 0.0)
        self.assertIn("srac_effective_candidates", diagnostics)
        self.assertIn("srac_node_mass_cv", diagnostics)

    def test_sr1p_reuses_frozen_p3_classifier_and_disables_replacement_head(self):
        model = P3SRACControlled(
            TinyP3(), "sr1p", cell_size=2, relation_channels=16
        )
        self.assertTrue(all(not p.requires_grad for p in model.p3.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in model.region_classifier.parameters())
        )
        self.assertTrue(model.structure_transition.requires_grad)
        contract = model.checkpoint_contract()
        self.assertTrue(contract["p3_damage_classifier_reused"])
        self.assertEqual(
            contract["prediction_unit"],
            "superpixel_feature_projection_to_p3_classifier",
        )

    def test_sr1p_transition_cannot_become_negative_high_pass(self):
        model = P3SRACControlled(
            TinyP3(), "sr1p", cell_size=2, relation_channels=16
        )
        with torch.no_grad():
            model.structure_transition.fill_(-2.0)
        outputs = model(self.rgb, self.sar, return_dict=True)
        self.assertEqual(float(outputs["srac_structure_transition"]), 0.0)

    def test_sr1pa_keeps_exact_p3_output_while_partition_gets_direct_gradient(self):
        p3 = TinyP3()
        model = P3SRACControlled(
            p3, "sr1pa", cell_size=2, relation_channels=16
        )
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )

        _loc, base_damage = p3(
            self.rgb, P3PseudoSARSource.to_p3_input(self.sar)
        )
        outputs = model(self.rgb, self.sar, return_dict=True)
        self.assertTrue(
            torch.allclose(
                base_damage.softmax(1), outputs["output_dam"].exp(), atol=1.0e-6
            )
        )
        self.assertFalse(model.structure_transition.requires_grad)
        labels = torch.randint(0, 4, (2, 32, 32))
        losses = structure_region_supervision(outputs, labels, 255, model=model)
        self.assertGreater(float(losses["representation_loss"]), 0.0)
        self.assertEqual(float(losses["srac_assignment_iterations"]), 1.0)
        losses["representation_loss"].backward()
        gradient = model.partition.embedding[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIn("srac_boundary_f1", losses)
        candidate_count = float(losses["srac_candidate_node_count"])
        if candidate_count > 0.0:
            expected = (
                float(losses["srac_background_leakage_sum"])
                / candidate_count
            )
            self.assertAlmostEqual(
                expected,
                float(losses["srac_background_leakage_fraction"]),
                places=6,
            )

    def test_iterative_variants_declare_three_assignment_rounds(self):
        pretrain = P3SRACControlled(
            TinyP3(), "sr1pai", cell_size=2, relation_channels=16
        )
        projection = P3SRACControlled(
            TinyP3(), "sr1pib", cell_size=2, relation_channels=16
        )
        self.assertEqual(pretrain.partition.assignment_iterations, 3)
        self.assertEqual(projection.partition.assignment_iterations, 3)
        self.assertTrue(pretrain.config.partition_pretrain)
        self.assertFalse(pretrain.structure_transition.requires_grad)
        self.assertFalse(projection.config.partition_pretrain)
        self.assertTrue(projection.structure_transition.requires_grad)
        self.assertEqual(pretrain.checkpoint_contract()["assignment_iterations"], 3)

        outputs = pretrain(self.rgb, self.sar, return_dict=True)
        losses = structure_region_supervision(
            outputs,
            torch.randint(0, 4, (2, 32, 32)),
            255,
            model=pretrain,
        )
        self.assertEqual(float(losses["srac_assignment_iterations"]), 3.0)
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )
        _loc, base_damage = pretrain.p3(
            self.rgb, P3PseudoSARSource.to_p3_input(self.sar)
        )
        self.assertTrue(
            torch.allclose(
                outputs["output_dam"].exp(),
                base_damage.softmax(1),
                atol=1.0e-6,
            )
        )

    def test_sr1paa_is_exact_p3_and_trains_only_affinity_head(self):
        p3 = TinyP3()
        model = P3SRACControlled(
            p3, "sr1paa", cell_size=2, relation_channels=16
        )
        from bda_benchmark.model.p3_reset.sar_sources.p3_pseudo_sar import (
            P3PseudoSARSource,
        )
        _loc, base_damage = p3(
            self.rgb, P3PseudoSARSource.to_p3_input(self.sar)
        )
        outputs = model(self.rgb, self.sar, return_dict=True)
        self.assertTrue(torch.equal(outputs["output_dam"], base_damage))
        self.assertEqual(outputs["srac_affinity_logits"].shape[1], 8)
        self.assertEqual(
            outputs["srac_affinity_logits"].shape,
            outputs["srac_affinity_valid"].shape,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.native_sar_encoder.parameters()
            )
        )
        labels = torch.randint(1, 4, (2, 32, 32))
        losses = structure_region_supervision(
            outputs, labels, 255, model=model
        )
        self.assertTrue(torch.isfinite(losses["representation_loss"]))
        losses["representation_loss"].backward()
        gradient = model.damage_affinity.pair_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIn("srac_affinity_damage_boundary_recall", losses)
        self.assertTrue(model.checkpoint_contract()["native_sar_encoder_frozen"])

    def test_sr1pa_partition_loss_is_bf16_autocast_safe(self):
        model = P3SRACControlled(
            TinyP3(), "sr1pa", cell_size=2, relation_channels=16
        )
        outputs = model(self.rgb, self.sar, return_dict=True)
        labels = torch.randint(0, 4, (2, 32, 32))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = structure_region_supervision(
                outputs, labels, 255, model=model
            )
        self.assertTrue(torch.isfinite(losses["representation_loss"]))

    def test_sr1pb_uses_boundary_partition_and_opens_only_safe_projection(self):
        model = P3SRACControlled(
            TinyP3(), "sr1pb", cell_size=2, relation_channels=16
        )
        self.assertTrue(model.config.boundary_aware_partition)
        self.assertFalse(model.config.partition_pretrain)
        self.assertTrue(model.structure_transition.requires_grad)
        model.set_optimizer_progress(50, 100)
        self.assertAlmostEqual(model.partition.effective_temperature, 0.325, places=5)
        outputs = model(self.rgb, self.sar, return_dict=True)
        self.assertEqual(float(outputs["srac_boundary_partition"]), 1.0)

    def test_advancement_audit_reads_bounded_sr1p_transition(self):
        negative = {
            "model": {"module.structure_transition": torch.tensor(-2.0)}
        }
        positive = {
            "model": {"module.structure_transition": torch.tensor(0.25)}
        }
        self.assertEqual(effective_structure_transition(negative), 0.0)
        observed = effective_structure_transition(positive)
        self.assertIsNotNone(observed)
        self.assertGreater(observed, 0.0)
        self.assertLess(observed, 1.0)

    def test_iterative_partition_gate_preserves_one_shot_geometry(self):
        baseline = {
            "srac_assignment_iterations": 3.0,
            "srac_partition_entropy": 0.94,
            "srac_effective_candidates": 7.9,
            "srac_background_leakage_fraction": 0.45,
            "srac_candidate_node_count": 100.0,
            "srac_building_iou": 0.48,
            "srac_boundary_f1": 0.01,
            "srac_partition_temperature": 0.50,
            "processed_images": 450,
        }
        candidate = dict(
            baseline,
            srac_partition_entropy=0.64,
            srac_effective_candidates=4.08,
            srac_background_leakage_fraction=0.24,
            srac_building_iou=0.6656,
            srac_boundary_f1=0.5789,
            srac_partition_temperature=0.15,
        )
        report = build_partition_report(baseline, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["next_step"],
            "PROCEED_TO_SR1PIB_SAFE_PROJECTION",
        )

        candidate["srac_boundary_f1"] = 0.57
        report = build_partition_report(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["boundary_f1_at_least_sr1pa_0_5789"]
        )

    def test_memory_is_class_balanced_and_read_does_not_mutate(self):
        memory = ClassBalancedRegionMemory(4, capacity_per_class=2)
        features = torch.randn(8, 4)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 2, 2])
        memory.enqueue(features, labels)
        self.assertEqual(memory.valid.sum(1).tolist(), [2, 1, 2])
        before = memory.cursor.clone()
        support, support_labels, valid = memory.retrieve(
            torch.randn(1, 3, 4), topk_per_class=1
        )
        self.assertEqual(support.shape, (1, 3, 3, 4))
        self.assertEqual(support_labels.shape, valid.shape)
        self.assertTrue(torch.equal(before, memory.cursor))

    def test_affinity_gate_requires_damage_boundary_evidence(self):
        baseline = {
            "srac_affinity_loss": 0.70,
            "srac_affinity_same_mean": 0.50,
            "srac_affinity_different_mean": 0.50,
            "srac_affinity_margin": 0.0,
            "srac_affinity_balanced_accuracy": 0.50,
            "srac_affinity_damage_boundary_recall": 0.50,
            "srac_affinity_valid_pairs": 100.0,
            "srac_affinity_same_count": 90.0,
            "srac_affinity_different_count": 10.0,
            "srac_affinity_damage_boundary_count": 5.0,
            "srac_affinity_negative_fraction": 0.10,
            "processed_images": 450,
        }
        candidate = dict(
            baseline,
            srac_affinity_loss=0.55,
            srac_affinity_same_mean=0.75,
            srac_affinity_different_mean=0.40,
            srac_affinity_margin=0.35,
            srac_affinity_balanced_accuracy=0.72,
            srac_affinity_damage_boundary_recall=0.68,
        )
        self.assertTrue(build_affinity_report(baseline, candidate)["passed"])
        candidate["srac_affinity_damage_boundary_recall"] = 0.59
        report = build_affinity_report(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["damage_boundary_recall_at_least_0_60"]
        )

    def test_rare_node_class_is_upweighted(self):
        weights = node_class_weights(torch.tensor([0, 0, 0, 1, 2, 2]))
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[2]), float(weights[0]))

    def test_partition_selector_uses_structure_not_unchanged_p3_metric(self):
        weaker = {
            "event_worst_official_style_4class_miou": 0.50,
            "srac_boundary_f1": 0.55,
            "srac_building_iou": 0.70,
            "srac_background_leakage_fraction": 0.20,
            "srac_partition_entropy": 0.60,
        }
        stronger = dict(
            weaker,
            srac_boundary_f1=0.60,
            # Ensure a worse classification tie cannot affect partition
            # selection when the P3 prediction is intentionally unchanged.
            event_worst_official_style_4class_miou=0.10,
        )
        self.assertGreater(
            checkpoint_selection_key(stronger, "srac_partition_quality"),
            checkpoint_selection_key(weaker, "srac_partition_quality"),
        )

    def test_affinity_selector_prioritises_balanced_accuracy(self):
        weaker = {
            "srac_affinity_balanced_accuracy": 0.65,
            "srac_affinity_damage_boundary_recall": 0.80,
            "srac_affinity_different_recall": 0.80,
            "srac_affinity_same_accuracy": 0.80,
        }
        stronger = dict(weaker, srac_affinity_balanced_accuracy=0.70)
        self.assertGreater(
            checkpoint_selection_key(
                stronger, "srac_affinity_balanced_accuracy"
            ),
            checkpoint_selection_key(
                weaker, "srac_affinity_balanced_accuracy"
            ),
        )


if __name__ == "__main__":
    unittest.main()
