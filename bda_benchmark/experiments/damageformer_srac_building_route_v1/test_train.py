"""CPU-only experiment-contract checks; no BRIGHT images are accessed."""
import argparse
import unittest

from .core import validate_budget
from .cross_q import crossed
from .evaluate import same_head_state
from .train import (assert_completed_audit, periodic_validation_with_capture_reset,
                    validate_arguments)
from .report import comparison


class RouteTrainContractTests(unittest.TestCase):
    @staticmethod
    def args(**overrides):
        values = dict(steps=2000, eval_every=500, lr=1e-4,
                      threshold=0.5, soft_floor=0.0, smoke=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_formal_budgets_and_smoke_are_separate(self):
        self.assertEqual(validate_budget(2000, False), 2000)
        self.assertEqual(validate_budget(5000, False), 5000)
        self.assertEqual(validate_budget(2, True), 2)
        for steps, smoke in ((2, False), (2000, True), (1999, False), (True, False)):
            with self.assertRaises(ValueError):
                validate_budget(steps, smoke)

    def test_periodic_curve_divides_full_budget(self):
        validate_arguments(self.args())
        validate_arguments(self.args(steps=5000, eval_every=500))
        with self.assertRaises(ValueError):
            validate_arguments(self.args(eval_every=600))

    def test_route_thresholds_reject_degenerate_or_invalid_values(self):
        for key, value in (("threshold", 0.0), ("threshold", 1.0),
                           ("soft_floor", -0.01), ("soft_floor", 0.5),
                           ("lr", 0.0)):
            with self.assertRaises(ValueError):
                validate_arguments(self.args(**{key: value}))

    def test_formal_audit_cannot_certify_failed_reproduction_or_prefix(self):
        report = dict(complete=True, image_count=450,
                      full_validation_evidence=True,
                      reproduction=dict(passed=True),
                      e0_metric_comparison=dict(passed=True),
                      q_a1_reproduction=dict(passed=True),
                      native_a1_replay_ok=True,
                      immutable={"files": True})
        assert_completed_audit(report, smoke=False)
        for key, changed in (("image_count", 449),
                             ("full_validation_evidence", False),
                             ("native_a1_replay_ok", False),
                             ("q_a1_reproduction", {"passed": False}),
                             ("e0_metric_comparison", {"passed": False}),
                             ("immutable", {"files": False})):
            bad = dict(report, **{key: changed})
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_completed_audit(bad, smoke=False)

    def test_cross_q_requires_matched_fit_and_reports_interaction(self):
        common = dict(protocol_sha256="e0", source_sha256="code", steps=2000,
                      threshold=.5, soft_floor=0., train_stream_sha256="same-images",
                      trainable_parameters={arm: 10 for arm in
                                            ("unsplit", "capacity_control", "hard", "soft")})
        original_fit = dict(common, q_source={"kind": "e0_original_q"})
        building_fit = dict(common, q_source={"kind": "completed_building_reconstruction_q"})
        def evaluation(fit, values):
            return dict(protocol_sha256=fit["protocol_sha256"],
                        source_sha256=fit["source_sha256"], q_source=fit["q_source"],
                        metrics={arm: {"valid_miou": score} for arm, score in values.items()})
        original_eval = evaluation(original_fit, dict(unsplit=.60, capacity_control=.605,
                                                       hard=.61, soft=.612))
        building_eval = evaluation(building_fit, dict(unsplit=.62, capacity_control=.625,
                                                       hard=.635, soft=.638))
        result = crossed(original_fit, original_eval, building_fit, building_eval)
        self.assertAlmostEqual(result["building_q_minus_original_q_pp"]["hard"], 2.5)
        self.assertAlmostEqual(result["q_route_interaction_pp"], .5)
        with self.assertRaises(ValueError):
            crossed(original_fit, original_eval,
                    dict(building_fit, train_stream_sha256="different"), building_eval)

    def test_report_contrasts_expected_four_arms(self):
        result = {"metrics": {arm: {"valid_miou": score,
                                    "iou": [0.9, score, score, score]}
                              for arm, score in (("unsplit", 0.60),
                                                 ("capacity_control", 0.61),
                                                 ("hard", 0.62), ("soft", 0.63))}}
        out = comparison(result)
        self.assertAlmostEqual(out["differences_pp"]["hard_minus_unsplit_pp"], 2.0)
        self.assertAlmostEqual(out["differences_pp"]["hard_minus_capacity_control_pp"], 1.0)
        self.assertAlmostEqual(out["differences_pp"]["soft_minus_hard_pp"], 1.0)

    def test_periodic_val_discards_calls_seen_by_training_q_hook(self):
        class Capture:
            calls = 0

            def clear(self):
                self.calls = 0

        capture = Capture()

        def validation():
            capture.calls += 450
            return {"image_count": 450}

        self.assertEqual(periodic_validation_with_capture_reset(
            validation, capture), {"image_count": 450})
        self.assertEqual(capture.calls, 0)
        capture.calls += 1
        self.assertEqual(capture.calls, 1)

        def failed_validation():
            capture.calls += 12
            raise RuntimeError("validation interrupted")

        with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
            periodic_validation_with_capture_reset(failed_validation, capture)
        self.assertEqual(capture.calls, 0)

    def test_readout_snapshot_comparison_normalizes_device_and_detects_change(self):
        import torch

        class OffCpuTensor:
            """Simulate a live tensor requiring transfer before torch.equal."""
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self.value

        snapshot = {"embedding": {"weight": torch.tensor([1.0, 2.0])}}
        current = {"embedding": {"weight": OffCpuTensor(torch.tensor([1.0, 2.0]))}}
        self.assertTrue(same_head_state(current, snapshot))
        current["embedding"]["weight"].value[1] = 3.0
        self.assertFalse(same_head_state(current, snapshot))
        self.assertFalse(same_head_state({"embedding": {}}, snapshot))


if __name__ == "__main__":
    unittest.main()
