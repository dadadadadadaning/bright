import unittest

import numpy as np
import torch

from bda_benchmark.script.standard_ML.spi_geometry_causal.core import (
    event_macro_paired_bootstrap,
    instance_labels_to_regions,
    local_window_gated_majority,
)


class GeometryCausalCoreTests(unittest.TestCase):
    def test_instance_labels_become_disjoint_regions(self):
        labels = np.array([[0, 1, 1], [2, 0, 2]], dtype=np.int32)
        regions = instance_labels_to_regions(labels, minimum_area=1)
        self.assertEqual([int(mask.sum()) for mask in regions], [2, 2])
        self.assertFalse(bool((regions[0] & regions[1]).any()))

    def test_local_control_does_not_change_outside_localization(self):
        raw = torch.tensor([[1, 1, 0], [1, 2, 0], [0, 0, 0]])
        probability = torch.full((4, 3, 3), 0.01)
        probability[1] = 0.90
        localization = torch.tensor(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        output, _ = local_window_gated_majority(
            raw,
            probability,
            localization,
            kernel_size=3,
            minimum_purity=0.5,
            minimum_margin=0.1,
        )
        self.assertEqual(int(output[1, 1]), 1)
        self.assertTrue(torch.equal(output[~localization], raw[~localization]))

    def test_low_margin_local_window_is_rejected(self):
        raw = torch.tensor([[1, 2], [1, 2]])
        probability = torch.full((4, 2, 2), 0.01)
        probability[1] = 0.45
        probability[2] = 0.44
        localization = torch.ones((2, 2), dtype=torch.bool)
        output, report = local_window_gated_majority(
            raw,
            probability,
            localization,
            kernel_size=3,
            minimum_purity=0.5,
            minimum_margin=0.1,
        )
        self.assertTrue(torch.equal(output, raw))
        self.assertEqual(report["accepted_pixels"], 0)

    def test_event_macro_bootstrap_gives_events_equal_weight(self):
        strong = [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 9]]
        weak = [[9, 0, 0, 0], [0, 1, 8, 0], [0, 0, 9, 0], [0, 0, 0, 9]]
        rows = [
            {
                "event": "small_event",
                "histograms": {"base": weak, "method": strong},
            }
        ]
        rows.extend(
            {
                "event": "large_event",
                "histograms": {"base": strong, "method": strong},
            }
            for _ in range(20)
        )
        report = event_macro_paired_bootstrap(
            rows,
            variants=("base", "method"),
            baseline="base",
            comparison_pairs=(),
            iterations=100,
            seed=1,
        )
        self.assertEqual(report["event_count"], 2)
        self.assertGreater(
            report["comparisons"]["method_minus_base"]["mean"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
