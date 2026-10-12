import unittest

import torch

from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    broadcast_labels,
    constant_label_region_audit,
    erode_region,
    foreground_majority_label,
    gated_majority_feedback,
    mean_probability_label,
    prediction_change_audit,
    predicted_building_regions,
    region_evidence,
)


class RegionFeedbackCoreTests(unittest.TestCase):
    def test_minimum_area_filters_small_components(self):
        loc = torch.zeros((8, 8), dtype=torch.long)
        loc[0, 0] = 1
        loc[3:5, 3:6] = 1
        regions = predicted_building_regions(loc, minimum_area=4)
        self.assertEqual(len(regions), 1)
        self.assertEqual(int(regions[0].sum().item()), 6)

    def test_regions_depend_only_on_localization(self):
        loc = torch.tensor([[1, 1, 0], [0, 0, 1]])
        regions = predicted_building_regions(loc, minimum_area=1)
        self.assertEqual(sorted(int(mask.sum()) for mask in regions), [1, 2])

    def test_majority_ignores_background_when_geometry_says_building(self):
        prediction = torch.tensor([[0, 0], [2, 1]])
        probabilities = torch.zeros((4, 2, 2))
        probabilities[2] = 0.6
        label = foreground_majority_label(
            prediction, probabilities, torch.ones((2, 2), dtype=torch.bool)
        )
        self.assertEqual(label, 2)

    def test_mean_probability_uses_foreground_classes(self):
        probabilities = torch.tensor(
            [
                [[0.90, 0.90]],
                [[0.02, 0.02]],
                [[0.03, 0.04]],
                [[0.05, 0.04]],
            ]
        )
        label = mean_probability_label(
            probabilities, torch.ones((1, 2), dtype=torch.bool)
        )
        self.assertEqual(label, 3)

    def test_broadcast_changes_only_region(self):
        base = torch.zeros((2, 3), dtype=torch.long)
        mask = torch.tensor([[False, True, True], [False, False, False]])
        output = broadcast_labels(base, [mask], [2])
        self.assertTrue(torch.equal(output, torch.tensor([[0, 2, 2], [0, 0, 0]])))

    def test_region_purity_penalizes_background_inside_localization(self):
        prediction = torch.tensor([[0, 0], [2, 2]])
        probabilities = torch.full((4, 2, 2), 0.05)
        probabilities[2] = 0.80
        evidence = region_evidence(
            prediction, probabilities, torch.ones((2, 2), dtype=torch.bool)
        )
        self.assertEqual(evidence.label, 2)
        self.assertAlmostEqual(evidence.purity, 0.5)

    def test_gate_rejects_low_purity_region(self):
        prediction = torch.tensor([[0, 0], [2, 2]])
        probabilities = torch.full((4, 2, 2), 0.05)
        probabilities[2] = 0.80
        output, records = gated_majority_feedback(
            prediction,
            probabilities,
            [torch.ones((2, 2), dtype=torch.bool)],
            minimum_purity=0.75,
            minimum_margin=0.10,
        )
        self.assertTrue(torch.equal(output, prediction))
        self.assertFalse(records[0]["gate_accepted"])

    def test_core_erosion_keeps_only_interior(self):
        mask = torch.ones((5, 5), dtype=torch.bool)
        core = erode_region(mask, radius=1)
        self.assertEqual(int(core.sum()), 9)
        self.assertTrue(bool(core[2, 2]))
        self.assertFalse(bool(core[0, 0]))

    def test_change_audit_separates_corrections_and_harms(self):
        base = torch.tensor([[1, 2, 3]])
        updated = torch.tensor([[2, 2, 1]])
        target = torch.tensor([[2, 2, 3]])
        audit = prediction_change_audit(base, updated, target)
        self.assertEqual(audit["changed_pixels"], 2)
        self.assertEqual(audit["corrected_pixels"], 1)
        self.assertEqual(audit["harmed_pixels"], 1)
        self.assertEqual(audit["net_gain_pixels"], 0)

    def test_constant_label_region_audit_matches_expected_changes(self):
        base = torch.tensor([[1, 2, 3]])
        target = torch.tensor([[2, 2, 3]])
        region = torch.ones((1, 3), dtype=torch.bool)
        write = torch.tensor([[True, True, False]])
        audit = constant_label_region_audit(base, target, region, write, label=2)
        self.assertEqual(audit["changed_pixels"], 1)
        self.assertEqual(audit["corrected_pixels"], 1)
        self.assertEqual(audit["harmed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
