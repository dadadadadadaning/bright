"""CPU contracts for strict, non-repairing failure diagnostics."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from .probability_check import (
    ProbabilityContractError, projection_diagnostics, validate_probability,
)


def _parent_projection():
    """Use the actual parent's pure projector without importing its backbone."""
    path = Path(__file__).parent.parent / "damageformer_srac_transformer_v1" / "model.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "project_region_probabilities")
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def _fixture():
    logits = torch.tensor([[[1., -1., 0., 2.], [3., 0., 2., -2.], [-2., 1., 4., 0.]]])
    q = torch.tensor([[[0.2, 0.5, 0.8, 0.0], [0.8, 0.5, 0.2, 1.0]]])
    ids = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    return logits, q, ids, (2, 2), (3, 5)


class ProbabilityContractTests(unittest.TestCase):
    def test_valid_probability_is_unmodified_and_summary_is_small(self):
        probability = torch.full((2, 4, 2, 3), .25)
        before = probability.clone()
        summary = validate_probability(probability, "region")
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["max_abs_class_sum_error"], 0)
        self.assertNotIn("worst_pixels", summary)
        self.assertTrue(torch.equal(probability, before))
        json.dumps(summary, allow_nan=False)

    def test_original_tolerance_accepts_small_honest_deviation(self):
        probability = torch.full((1, 4, 1, 2), .25) * (1 + 3e-5)
        summary = validate_probability(probability, "small_deviation")
        self.assertTrue(summary["passed"])
        self.assertGreater(summary["max_abs_class_sum_error"], 2e-5)

    def test_above_original_tolerance_is_not_repaired(self):
        for multiplier in (1 + 5e-5, 1.1, .5, 0.):
            with self.subTest(multiplier=multiplier):
                probability = torch.full((1, 4, 2, 3), .25) * multiplier
                before = probability.clone()
                with self.assertRaises(ProbabilityContractError) as raised:
                    validate_probability(probability, "edge_zero")
                self.assertIn("not normalized: edge_zero", str(raised.exception))
                evidence = raised.exception.diagnostics
                self.assertEqual(evidence["reasons"], ["class_sum_not_one"])
                self.assertLessEqual(len(evidence["worst_pixels"]), 4)
                self.assertEqual(set(evidence["class_sums"]), {"native", "cpu_float32", "cpu_float64"})
                self.assertTrue(torch.equal(probability, before))
                json.dumps(evidence, allow_nan=False)

    def test_nan_infinities_and_negative_values_fail_with_json_safe_details(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -.1):
            with self.subTest(bad=bad):
                probability = torch.full((1, 4, 2, 3), .25)
                probability[0, 1, 1, 2] = bad
                with self.assertRaises(ProbabilityContractError) as raised:
                    validate_probability(probability, "edge_zero")
                evidence = raised.exception.diagnostics
                self.assertEqual(evidence["worst_pixels"][0]["coordinate_bhw"], [0, 1, 2])
                self.assertGreater(evidence["negative_count"] + evidence["nonfinite_count"], 0)
                json.dumps(evidence, allow_nan=False)

    def test_negative_class_rejected_even_when_sum_is_exactly_one(self):
        probability = torch.tensor([[-.1, .4, .3, .4]]).reshape(1, 4, 1, 1)
        with self.assertRaises(ProbabilityContractError) as raised:
            validate_probability(probability, "negative")
        self.assertIn("negative_values", raised.exception.diagnostics["reasons"])
        self.assertNotIn("class_sum_not_one", raised.exception.diagnostics["reasons"])

    def test_invalid_shapes_and_non_tensor_are_explicit_failures(self):
        for value in (None, torch.empty(0, 4, 1, 1), torch.ones(1, 3, 2, 2), torch.ones(1, 4)):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ProbabilityContractError) as raised:
                    validate_probability(value, "bad_shape")
                json.dumps(raised.exception.diagnostics, allow_nan=False)


class ProjectionDiagnosticsTests(unittest.TestCase):
    def test_replays_match_actual_parent_and_preserve_inputs(self):
        inputs = _fixture()
        saved = [value.clone() for value in inputs[:3]]
        observed = _parent_projection()(*inputs)
        before = observed.clone()
        report = projection_diagnostics(*inputs, observed)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["predictions_replaced"])
        for replay in report["replays"].values():
            self.assertEqual(replay["status"], "completed")
            self.assertTrue(replay["probability_contract"]["passed"])
            self.assertTrue(replay["comparison_to_observed"]["exactly_equal"])
            self.assertEqual(replay["comparison_to_observed"]["max_abs_difference"], 0)
            self.assertEqual(replay["stages"]["class_denominator"]["below_floor_count"], 0)
            self.assertEqual(set(replay["stages"]), {
                "softmax", "sparse_q_mixture", "bilinear", "class_denominator", "normalized_result",
            })
        self.assertTrue(torch.equal(observed, before))
        for value, prior in zip(inputs[:3], saved):
            self.assertTrue(torch.equal(value, prior))
        json.dumps(report, allow_nan=False)

    def test_replay_distinguishes_observed_output_from_same_formula_result(self):
        inputs = _fixture()
        observed = _parent_projection()(*inputs) * .9
        report = projection_diagnostics(*inputs, observed)
        for replay in report["replays"].values():
            self.assertFalse(replay["comparison_to_observed"]["allclose_at_original_tolerance"])
            self.assertGreater(replay["comparison_to_observed"]["max_abs_difference"], .01)
            self.assertTrue(replay["probability_contract"]["passed"])

    def test_denominator_floor_is_observed_without_repair(self):
        logits, q, ids, shape, output_size = _fixture()
        q = q * 1e-10
        observed = _parent_projection()(logits, q, ids, shape, output_size)
        report = projection_diagnostics(logits, q, ids, shape, output_size, observed)
        for replay in report["replays"].values():
            self.assertEqual(replay["stages"]["class_denominator"]["below_floor_count"], 15)
            self.assertFalse(replay["probability_contract"]["passed"])
            self.assertTrue(replay["comparison_to_observed"]["exactly_equal"])

    def test_each_backend_error_is_reported_and_diagnostics_are_json_safe(self):
        logits, q, ids, shape, output_size = _fixture()
        ids[0, 0] = 1000
        observed = torch.full((1, 4, *output_size), float("nan"))
        report = projection_diagnostics(logits, q, ids, shape, output_size, observed)
        self.assertEqual(report["sources"]["observed"]["nan_count"], 60)
        for replay in report["replays"].values():
            self.assertEqual(replay["status"], "error")
            self.assertIn("error_type", replay)
            self.assertIn("softmax", replay["stages"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
