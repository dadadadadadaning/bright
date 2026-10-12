"""Failure evidence must preserve the exact invalid output and fail closed."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import probability_prediction
from .evaluate import select_validation_images
from .probability_diagnostics import DiagnosticProbabilityError, checked_predictions


class ProbabilityDiagnosticContracts(unittest.TestCase):
    def test_valid_values_and_predictions_are_not_modified(self):
        p = torch.tensor([[[[0.1]], [[0.5]], [[0.3]], [[0.1]]]])
        before = p.clone()
        with tempfile.TemporaryDirectory() as folder:
            prediction = checked_predictions({"p": p}, ["p"], (1, 1),
                validator=probability_prediction, image="synthetic", output=folder)
            self.assertEqual(int(prediction["p"].item()), 1)
            self.assertTrue(torch.equal(p, before))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_nonunit_probability_captures_arm_raw_tensor_and_never_continues(self):
        valid = torch.full((1, 4, 2, 2), 0.25)
        invalid = valid.clone()
        invalid[0, 2, 1, 0] += 0.001
        original = invalid.clone()
        calls = []

        def validator(value, size):
            calls.append(value)
            return probability_prediction(value, size)

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(DiagnosticProbabilityError, "arm=r"):
                checked_predictions({"p": valid, "r": invalid, "region_mean": valid, "t": valid},
                    ["p", "r", "region_mean", "t"], (2, 2), validator=validator,
                    image="morocco-earthquake_00000194", output=folder,
                    context={"q": torch.ones(1, 1, 4)})
            self.assertEqual(len(calls), 2)
            self.assertTrue(torch.equal(original, invalid))
            report = json.loads((Path(folder) / "probability_failure.json").read_text())
            self.assertEqual(report["arm"], "r")
            self.assertFalse(report["full_validation_evidence"])
            self.assertEqual(report["normalization_atol"], 2e-5)
            self.assertEqual(report["historical_metric_tolerance"], 1e-5)
            self.assertEqual(report["arms"]["r"]["cpu_float64_sum"]["outside_tolerance"], 1)
            snapshot = torch.load(Path(folder) / "probability_failure.pt", weights_only=True)
            self.assertTrue(torch.equal(snapshot["probabilities"]["r"], original))
            self.assertIn("region_mean", snapshot["probabilities"])
            self.assertNotIn("target", snapshot)

    def test_nonfinite_output_remains_failure_and_json_has_no_nan(self):
        p = torch.full((1, 4, 1, 1), 0.25)
        p[0, 0, 0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(DiagnosticProbabilityError, "Invalid probabilities"):
                checked_predictions({"j3": p}, ["j3"], (1, 1), validator=probability_prediction,
                                    image="synthetic", output=folder)
            raw = (Path(folder) / "probability_failure.json").read_text()
            self.assertNotIn("NaN", raw)
            self.assertFalse(json.loads(raw)["arms"]["j3"]["finite"])

    def test_single_image_can_never_be_selected_as_full_validation(self):
        names = [f"synthetic_{i:08d}" for i in range(450)]
        selected, diagnostic = select_validation_images(names, image=names[79])
        self.assertEqual(selected, [names[79]])
        self.assertTrue(diagnostic)
        self.assertEqual(select_validation_images(names), (names, False))
        for image, limit in (("not_in_val", 0), (names[79], 2)):
            with self.assertRaises(ValueError):
                select_validation_images(names, limit=limit, image=image)


if __name__ == "__main__":
    unittest.main()
