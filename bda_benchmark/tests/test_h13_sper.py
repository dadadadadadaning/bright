import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    factorize_building_logits,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (
    H13_VARIANTS,
    BoundaryGatedRegionGraph,
    DamageFormerRGBSARH13SPER,
    RGBGuidedSoftSuperpixels,
    SuperpixelEvidenceStage,
    pool_superpixel_statistics,
    unpool_superpixel_values,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
)
from bda_benchmark.script.standard_ML.train_H13_SPER import (
    configure_h13_freeze_policy,
)
from bda_benchmark.script.standard_ML.eval_H13_SPER import advancement_gate
from bda_benchmark.script.standard_ML.audit_H13_superpixels import (
    decide_partition_route,
)


class H13SPERTests(unittest.TestCase):
    def test_variants_are_a_controlled_chain(self):
        self.assertEqual(tuple(H13_VARIANTS), ("s0", "s1", "s2", "s3"))
        self.assertEqual(H13_VARIANTS["s1"].assignment_mode, "grid")
        self.assertEqual(H13_VARIANTS["s2"].assignment_mode, "rgb")
        self.assertFalse(H13_VARIANTS["s2"].graph_refinement)
        self.assertTrue(H13_VARIANTS["s3"].graph_refinement)

    def test_s0_has_exact_h6_a2_state_contract(self):
        torch.manual_seed(13)
        source = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
        torch.manual_seed(13)
        target = DamageFormerRGBSARH13SPER(variant="s0", pretrained_rgb=False)
        self.assertEqual(set(source.state_dict()), set(target.state_dict()))
        target.load_state_dict(source.state_dict(), strict=True)

    def test_soft_assignments_are_normalized_and_local(self):
        module = RGBGuidedSoftSuperpixels(cell_size=2)
        assignment = module(torch.randn(2, 3, 7, 9), mode="rgb")
        self.assertEqual(tuple(assignment.weights.shape), (2, 9, 63))
        self.assertTrue(
            torch.allclose(
                assignment.weights.sum(1),
                torch.ones(2, 63),
                atol=1.0e-6,
            )
        )
        self.assertTrue(torch.isfinite(assignment.entropy).all())
        self.assertGreaterEqual(float(assignment.entropy.min()), 0.0)
        self.assertLessEqual(float(assignment.entropy.max()), 1.0 + 1.0e-6)

    def test_constant_region_pool_and_unpool_are_identity(self):
        module = RGBGuidedSoftSuperpixels(cell_size=2)
        rgb = torch.randn(1, 3, 6, 6)
        assignment = module(rgb, mode="rgb")
        feature = torch.full((1, 4, 6, 6), 3.0)
        mean, std, maximum, mass = pool_superpixel_statistics(feature, assignment)
        self.assertTrue(torch.allclose(mean[mass.expand_as(mean) > 0], torch.tensor(3.0)))
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.allclose(maximum[mass.expand_as(maximum) > 0], torch.tensor(3.0)))
        restored = unpool_superpixel_values(mean, assignment, (6, 6))
        self.assertTrue(torch.allclose(restored, feature, atol=1.0e-5))

    def test_grid_and_rgb_modes_are_not_the_same_partition(self):
        module = RGBGuidedSoftSuperpixels(cell_size=2, color_temperature=0.1)
        rgb = torch.zeros(1, 3, 6, 6)
        rgb[:, :, :, 3:] = 5.0
        grid = module(rgb, mode="grid")
        guided = module(rgb, mode="rgb")
        self.assertFalse(torch.equal(grid.weights, guided.weights))

    def test_region_head_is_exactly_zero_but_trainable(self):
        head = SuperpixelEvidenceStage(
            source_channels=8,
            hidden_channels=8,
            assignment_mode="rgb",
            cell_size=2,
        )
        rgb_feature = torch.randn(2, 8, 8, 8)
        sar_feature = torch.randn_like(rgb_feature)
        pre_rgb = torch.randn(2, 3, 32, 32)
        support = torch.ones(2, 1, 32, 32)
        occurrence = torch.randn(2, 1, 32, 32)
        output, diagnostics = head(
            rgb_feature, sar_feature, pre_rgb, support, occurrence
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertIn("h13_assignment_entropy", diagnostics)
        output.square().mean().backward()
        self.assertIsNotNone(head.output.weight.grad)

    def test_superpixel_residual_is_blocked_outside_buildings(self):
        head = SuperpixelEvidenceStage(
            source_channels=8,
            hidden_channels=8,
            assignment_mode="grid",
            cell_size=2,
        )
        with torch.no_grad():
            head.output.bias.fill_(1.0)
        output, _ = head(
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 3, 32, 32),
            torch.zeros(1, 1, 32, 32),
            torch.randn(1, 1, 32, 32),
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_graph_step_is_finite_and_safe_at_initialization(self):
        graph = BoundaryGatedRegionGraph(8)
        nodes = torch.randn(2, 12, 8, requires_grad=True)
        rgb = torch.randn(2, 12, 3)
        support = torch.rand(2, 12, 1)
        output, update = graph(nodes, rgb, support, 3, 4)
        self.assertTrue(torch.equal(output, nodes))
        self.assertTrue(torch.equal(update, torch.zeros_like(update)))
        output.square().mean().backward()
        self.assertIsNotNone(graph.transition.grad)
        self.assertTrue(torch.isfinite(graph.transition.grad))

    def test_s2_forward_is_exact_a2_at_safe_initialization(self):
        model = DamageFormerRGBSARH13SPER(variant="s2", pretrained_rgb=False)
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 1, 64, 64),
                return_dict=True,
            )
        self.assertTrue(torch.equal(outputs["output_dam"], outputs["native_output_dam"]))
        self.assertIn("h13_assignment_entropy", outputs)

    def test_occurrence_update_preserves_severity_odds(self):
        model = DamageFormerRGBSARH13SPER(variant="s2", pretrained_rgb=False)
        configure_h13_freeze_policy(model)
        with torch.no_grad():
            model.region_head.output.bias.fill_(0.5)
            model.affected_transition.fill_(1.0)
        model.eval()
        outputs = model(
            torch.randn(1, 3, 64, 64),
            torch.randn(1, 1, 64, 64),
            return_dict=True,
        )
        _base_occurrence, base_severity = factorize_building_logits(
            outputs["native_output_dam"]
        )
        _occurrence, severity = factorize_building_logits(outputs["output_dam"])
        self.assertTrue(torch.allclose(severity, base_severity, atol=2.0e-5))

    def test_freeze_policy_trains_only_region_modules(self):
        model = DamageFormerRGBSARH13SPER(variant="s3", pretrained_rgb=False)
        configure_h13_freeze_policy(model)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith(("region_head", "affected_transition")) for name in trainable)
        )

    def test_advancement_gate_uses_real_event_deltas(self):
        baseline = {
            "damaged_iou": 0.40,
            "affected_recall": 0.70,
            "damaged_gate_miss": 0.35,
            "event_mean_official_style_4class_miou": 0.50,
            "event_worst_official_style_4class_miou": 0.40,
            "miou4": 0.66,
        }
        observed = dict(baseline)
        observed.update(
            damaged_iou=0.404,
            affected_recall=0.706,
            damaged_gate_miss=0.344,
            event_mean_official_style_4class_miou=0.501,
            event_worst_official_style_4class_miou=0.401,
            miou4=0.661,
        )
        old_events = {f"e{i}": {"miou4": 0.50} for i in range(14)}
        new_events = {
            f"e{i}": {"miou4": 0.51 if i < 10 else 0.49}
            for i in range(14)
        }
        gate = advancement_gate(observed, baseline, new_events, old_events)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["observed_pp"]["nonnegative_events"], 10)

    def test_partition_decision_refuses_empty_smoke_support(self):
        record = {
            "affected_pixels": 0,
            "best_fraction_oracle": {"affected_iou": 0.0},
            "h6_a2_affected_iou_at_audit_scale": 0.0,
        }
        decision = decide_partition_route({"grid": record, "rgb": record})
        self.assertEqual(
            decision, "INSUFFICIENT_AFFECTED_SUPPORT_RUN_FULL_AUDIT"
        )


if __name__ == "__main__":
    unittest.main()
