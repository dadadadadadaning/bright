import unittest

import numpy as np
import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (
    apply_hierarchical_residuals,
    factorize_building_logits,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H12_DGB import (
    H12_VARIANTS,
    OCCURRENCE_CANDIDATE_BOUND,
    OCCURRENCE_TRANSITION_MAX,
    DamageFormerRGBSARH12DGB,
    WeakDamageBypassStage,
    bound_occurrence_candidate,
    bounded_occurrence_transition,
    erode_building_support,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
)
from bda_benchmark.script.standard_ML.audit_H12_shallow_evidence import decide
from bda_benchmark.script.standard_ML.h12_dgb_runtime import (
    EventBalancedSampler,
    binary_auprc,
    event_protocol_key,
    fn_aware_hierarchical_nll,
    hierarchical_damage_nll,
    parameter_free_difference_scores,
    update_h12_selection_records,
)
from bda_benchmark.script.standard_ML.train_H12_DGB import (
    configure_h12_freeze_policy,
)


class _TinyDataset:
    def __init__(self) -> None:
        self.data_list = [
            "event-a_00000000",
            "event-a_00000001",
            "event-b_00000000",
        ]
        self.split = "train"


class H12DGBTests(unittest.TestCase):
    def test_variants_form_the_controlled_chain(self):
        self.assertEqual(tuple(H12_VARIANTS), ("d0", "d1", "d2", "d3"))
        self.assertEqual(H12_VARIANTS["d1"].bypass_stages, ("stage2",))
        self.assertEqual(H12_VARIANTS["d2"].bypass_stages, ("stage2", "stage3"))
        self.assertTrue(H12_VARIANTS["d3"].fn_aware_occurrence)
        self.assertFalse(H12_VARIANTS["d1"].fn_aware_occurrence)

    def test_d0_has_exact_h6_a2_state_contract(self):
        torch.manual_seed(13)
        source = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
        torch.manual_seed(13)
        target = DamageFormerRGBSARH12DGB(variant="d0", pretrained_rgb=False)
        self.assertEqual(set(source.state_dict()), set(target.state_dict()))
        target.load_state_dict(source.state_dict(), strict=True)

    def test_eroded_building_support_blocks_background(self):
        building = torch.zeros(1, 1, 7, 7)
        building[:, :, 2:5, 2:5] = 1.0
        eroded = erode_building_support(building, radius=1, threshold=0.5)
        self.assertEqual(float(eroded[0, 0, 3, 3]), 1.0)
        self.assertEqual(float(eroded[0, 0, 2, 2]), 0.0)
        self.assertEqual(float(eroded[0, 0, 0, 0]), 0.0)

    def test_bypass_is_zero_outside_building_roi(self):
        head = WeakDamageBypassStage(8, hidden_channels=8)
        rgb = torch.randn(2, 8, 6, 6)
        sar = torch.randn_like(rgb)
        support = torch.zeros(2, 1, 6, 6)
        residual, diagnostics = head(rgb, sar, support)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(
            torch.equal(diagnostics["h12_building_support"], torch.zeros_like(residual))
        )

    def test_d1_forward_is_exact_a2_at_safe_initialization(self):
        model = DamageFormerRGBSARH12DGB(variant="d1", pretrained_rgb=False)
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 1, 64, 64),
                return_dict=True,
            )
        self.assertTrue(torch.equal(outputs["output_dam"], outputs["native_output_dam"]))
        self.assertEqual(tuple(outputs["h10_affected_logits"].shape), (1, 1, 64, 64))

    def test_occurrence_candidate_bound_keeps_finite_severity(self):
        huge = torch.full((2, 1, 8, 8), 1.0e5)
        bounded = bound_occurrence_candidate(huge)
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertLessEqual(float(bounded.abs().max()), OCCURRENCE_CANDIDATE_BOUND)
        residual = torch.tanh(torch.tensor(1.0)) * bounded
        logits = torch.randn(2, 4, 8, 8)
        output = apply_hierarchical_residuals(logits, occurrence_residual=residual)
        occurrence, severity = factorize_building_logits(output)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(occurrence).all())
        self.assertTrue(torch.isfinite(severity).all())
        self.assertTrue(
            torch.allclose(severity, logits[:, 3:4] - logits[:, 2:3], atol=2e-5)
        )

    def test_occurrence_candidate_rejects_nonfinite_input(self):
        with self.assertRaises(FloatingPointError):
            bound_occurrence_candidate(torch.tensor([float("nan")]))

    def test_bounded_transition_rejects_nonfinite_parameter(self):
        with self.assertRaises(FloatingPointError):
            bounded_occurrence_transition(torch.tensor(float("nan")))
        value = bounded_occurrence_transition(torch.tensor(0.0))
        self.assertGreater(float(value), 0.0)
        self.assertLess(float(value), OCCURRENCE_TRANSITION_MAX)

    def test_constant_local_moments_have_finite_backward(self):
        value = torch.ones(2, 8, 6, 6, requires_grad=True)
        mean, std = WeakDamageBypassStage._local_moments(value)
        (mean.mean() + std.mean()).backward()
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_d1_stays_finite_after_transition_leaves_zero(self):
        model = DamageFormerRGBSARH12DGB(variant="d1", pretrained_rgb=False)
        configure_h12_freeze_policy(model)
        model.affected_transition.data.fill_(1.0)
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 3, 64, 64) * 8,
                torch.randn(1, 1, 64, 64) * 8,
                return_dict=True,
            )
        self.assertTrue(torch.isfinite(outputs["output_dam"]).all())
        self.assertTrue(torch.isfinite(outputs["h10_affected_logits"]).all())
        self.assertTrue(torch.isfinite(outputs["h10_severity_logits"]).all())
        labels = torch.randint(0, 4, (1, 64, 64))
        loss = hierarchical_damage_nll(outputs, labels, 255)
        self.assertTrue(torch.isfinite(loss["extra_supervised_loss"]))

    def test_zero_output_head_is_exact_but_receives_first_step_gradient(self):
        head = WeakDamageBypassStage(8, hidden_channels=8)
        transition = torch.nn.Parameter(torch.tensor(0.0))
        rgb = torch.randn(2, 8, 6, 6)
        sar = torch.randn_like(rgb)
        support = torch.ones(2, 1, 6, 6)
        candidate, _ = head(rgb, sar, support)
        base = torch.randn(2, 4, 6, 6)
        residual = bounded_occurrence_transition(transition) * candidate
        output = apply_hierarchical_residuals(base, occurrence_residual=residual)
        self.assertTrue(torch.equal(output, base))
        output.square().mean().backward()
        self.assertIsNotNone(transition.grad)
        self.assertTrue(torch.isfinite(transition.grad))
        self.assertIsNotNone(head.output.weight.grad)
        self.assertTrue(torch.isfinite(head.output.weight.grad).all())
        self.assertGreater(float(head.output.weight.grad.abs().sum()), 0.0)

    def test_bypass_remains_finite_across_optimizer_steps(self):
        torch.manual_seed(23)
        head = WeakDamageBypassStage(8, hidden_channels=8)
        transition = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW(
            [*head.parameters(), transition], lr=1.0e-3
        )
        rgb = torch.ones(2, 8, 6, 6)
        sar = torch.ones_like(rgb)
        support = torch.ones(2, 1, 6, 6)
        target = torch.full((2, 1, 6, 6), 0.1)

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            candidate, _ = head(rgb, sar, support)
            residual = bounded_occurrence_transition(
                transition
            ) * bound_occurrence_candidate(candidate)
            loss = (residual - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            parameters = [
                parameter
                for parameter in [*head.parameters(), transition]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters, 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            for parameter in [*head.parameters(), transition]:
                self.assertTrue(torch.isfinite(parameter).all())

    def test_occurrence_residual_preserves_severity_odds(self):
        logits = torch.randn(2, 4, 8, 8)
        residual = torch.randn(2, 1, 8, 8)
        output = apply_hierarchical_residuals(logits, occurrence_residual=residual)
        self.assertTrue(
            torch.allclose(output[:, 3] - output[:, 2], logits[:, 3] - logits[:, 2], atol=2e-6)
        )
        self.assertTrue(torch.equal(output[:, 0], logits[:, 0]))
        self.assertTrue(torch.equal(output[:, 1], logits[:, 1]))

    def test_freeze_policy_trains_only_the_bypass(self):
        model = DamageFormerRGBSARH12DGB(variant="d2", pretrained_rgb=False)
        configure_h12_freeze_policy(model)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith(("bypass_heads", "affected_transition")) for name in trainable)
        )

    def test_frozen_a2_is_not_in_the_backward_graph(self):
        model = DamageFormerRGBSARH12DGB(variant="d1", pretrained_rgb=False)
        configure_h12_freeze_policy(model)
        with torch.no_grad():
            model.affected_transition.fill_(0.25)
        outputs = model(
            torch.randn(1, 3, 64, 64),
            torch.randn(1, 1, 64, 64),
            return_dict=True,
        )
        self.assertFalse(outputs["stage2_rgb_common"].requires_grad)
        self.assertFalse(outputs["native_output_dam"].requires_grad)
        self.assertTrue(outputs["output_dam"].requires_grad)
        outputs["h10_affected_logits"].float().mean().backward()
        self.assertIsNotNone(model.affected_transition.grad)
        self.assertIsNotNone(model.bypass_heads["stage2"].output.weight.grad)
        leaked = [
            name
            for name, value in model.named_parameters()
            if value.grad is not None
            and not name.startswith(("bypass_heads", "affected_transition"))
        ]
        self.assertEqual(leaked, [])

    def test_fn_aware_loss_replaces_occurrence_and_ignores_background(self):
        occurrence = torch.zeros(1, 1, 2, 2, requires_grad=True)
        severity = torch.zeros(1, 1, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 1], [2, 3]]])
        outputs = {
            "h10_affected_logits": occurrence,
            "h10_severity_logits": severity,
        }
        baseline = hierarchical_damage_nll(outputs, labels, 255)
        fn_aware = fn_aware_hierarchical_nll(
            outputs, labels, 255, fn_weight=2.0, fp_weight=1.0, focal_gamma=0.0
        )
        self.assertGreater(
            float(fn_aware["extra_supervised_loss"]),
            float(baseline["extra_supervised_loss"]),
        )
        fn_aware["extra_supervised_loss"].backward()
        self.assertIsNotNone(occurrence.grad)

    def test_checkpoint_protocol_prefers_event_worst(self):
        records = {}
        weak_worst = {
            "event_worst_official_style_4class_miou": 0.10,
            "event_mean_official_style_4class_miou": 0.50,
            "damaged_iou": 0.40,
            "valid_miou": 0.60,
            "damage_balance_score": 0.0,
        }
        strong_worst = dict(weak_worst)
        strong_worst["event_worst_official_style_4class_miou"] = 0.11
        strong_worst["valid_miou"] = 0.50
        update_h12_selection_records(records, weak_worst, 100)
        improved = update_h12_selection_records(records, strong_worst, 200)
        self.assertIn("valid", improved)
        self.assertEqual(records["valid"]["step"], 200)
        self.assertGreater(event_protocol_key(strong_worst), event_protocol_key(weak_worst))

    def test_binary_auprc_and_difference_scores(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        self.assertAlmostEqual(binary_auprc(np.asarray([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
        self.assertIsNone(binary_auprc(np.ones(4), np.ones(4, dtype=np.uint8)))
        scores = parameter_free_difference_scores(
            torch.ones(1, 4, 3, 3), torch.zeros(1, 4, 3, 3)
        )
        self.assertGreater(float(scores["abs_diff_mean"].mean()), 0.0)
        self.assertGreater(float(scores["cosine_distance"].mean()), 0.0)

    def test_event_balanced_sampler_covers_every_event(self):
        sampler = EventBalancedSampler(_TinyDataset(), generator=torch.Generator().manual_seed(0))
        drawn = [index for index, _ in zip(sampler, range(30))]
        events = {_TinyDataset().data_list[index].rsplit("_", 1)[0] for index in drawn}
        self.assertEqual(events, {"event-a", "event-b"})

    def test_audit_decision_prefers_a2_shallow_bypass(self):
        tasks = {
            "damaged_gate_miss": {
                "a2_stage2_abs_diff_mean": {
                    "auroc": 0.62,
                    "event_macro_auroc": 0.60,
                    "event_worst_auroc": 0.52,
                    "comparable_events": 12,
                    "event_direction_consistent": 10,
                },
                "sarmae_block8_energy": {
                    "auroc": 0.51,
                    "event_macro_auroc": 0.50,
                    "event_worst_auroc": 0.40,
                    "comparable_events": 12,
                    "event_direction_consistent": 6,
                },
            }
        }
        decision = decide(tasks)
        self.assertEqual(decision["recommendation"], "IMPLEMENT_H12_D1_STAGE2_BYPASS")


if __name__ == "__main__":
    unittest.main()
