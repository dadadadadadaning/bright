import unittest

import numpy as np
import torch

from bda_benchmark.script.standard_ML.audit_CCU_local_correspondence import (
    METRIC_NAMES,
    RunningMoments,
    binary_auroc,
    cohen_d,
    disaster_type,
    parameter_free_local_correspondence,
    pure_stage_labels,
)


class CCULocalCorrespondenceAuditTests(unittest.TestCase):
    def test_running_moments_ignores_nonfinite_values(self):
        moments = RunningMoments()
        moments.update(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
        result = moments.summary()
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mean"], 1.5)
        self.assertAlmostEqual(result["std"], 0.5)

    def test_cohen_d_preserves_direction(self):
        first = {"count": 10, "mean": 0.0, "std": 1.0}
        second = {"count": 10, "mean": 1.0, "std": 1.0}
        self.assertAlmostEqual(cohen_d(first, second), 1.0)
        self.assertAlmostEqual(cohen_d(second, first), -1.0)

    def test_binary_auroc_handles_perfect_reverse_and_ties(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        self.assertAlmostEqual(
            binary_auroc(np.asarray([0.0, 0.1, 0.8, 0.9]), labels), 1.0
        )
        self.assertAlmostEqual(
            binary_auroc(np.asarray([0.9, 0.8, 0.1, 0.0]), labels), 0.0
        )
        self.assertAlmostEqual(binary_auroc(np.ones(4), labels), 0.5)
        self.assertIsNone(binary_auroc(np.ones(4), np.ones(4, dtype=np.uint8)))

    def test_parameter_free_matcher_returns_only_declared_e0_statistics(self):
        torch.manual_seed(3407)
        diagnostics = parameter_free_local_correspondence(
            torch.randn(2, 8, 7, 7),
            torch.randn(2, 8, 7, 7),
            window_size=3,
        )
        for key in METRIC_NAMES:
            self.assertIn(key, diagnostics)
            self.assertEqual(tuple(diagnostics[key].shape), (2, 1, 7, 7))
            self.assertTrue(torch.isfinite(diagnostics[key]).all())
        self.assertNotIn("cycle_error", diagnostics)
        self.assertNotIn("cycle_consistency", diagnostics)
        self.assertGreaterEqual(float(diagnostics["correlation_peak"].min()), -1.0)
        self.assertLessEqual(float(diagnostics["correlation_peak"].max()), 1.0)
        self.assertGreaterEqual(float(diagnostics["correlation_entropy"].min()), 0.0)
        self.assertLessEqual(float(diagnostics["correlation_entropy"].max()), 1.0)

    def test_peak_offset_recovers_a_known_one_cell_shift(self):
        # At the centre, RGB channel 0 matches SAR one cell to the right.
        rgb = torch.zeros(1, 2, 5, 5)
        sar = torch.zeros_like(rgb)
        rgb[0, 0, 2, 2] = 1.0
        sar[0, 0, 2, 3] = 1.0
        diagnostics = parameter_free_local_correspondence(
            rgb, sar, window_size=3, temperature=0.1
        )
        self.assertAlmostEqual(
            float(diagnostics["correlation_peak"][0, 0, 2, 2]), 1.0
        )
        self.assertAlmostEqual(
            float(diagnostics["offset_magnitude"][0, 0, 2, 2]), 1.0
        )

    def test_stage_labels_reject_mixed_cells(self):
        damage = torch.tensor(
            [[[1, 1, 2, 2], [1, 1, 2, 2], [1, 2, 3, 3], [2, 1, 3, 3]]]
        )
        labels = pure_stage_labels(damage, (2, 2), min_class_purity=0.75)
        self.assertEqual(int(labels[0, 0, 0]), 1)
        self.assertEqual(int(labels[0, 0, 1]), 2)
        self.assertEqual(int(labels[0, 1, 0]), 255)
        self.assertEqual(int(labels[0, 1, 1]), 3)

    def test_disaster_type_uses_event_suffix(self):
        self.assertEqual(disaster_type("la_palma-volcano_00000001"), "volcano")
        self.assertEqual(disaster_type("noto-earthquake"), "earthquake")


if __name__ == "__main__":
    unittest.main()
