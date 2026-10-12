import unittest

import torch

from bda_benchmark.script.standard_ML.spi_t1.bag_audit import (
    finalize_counts,
    new_counts,
    predicted_localization_bags,
    semantic_component_bags,
    update_region_counts,
)


class MICBBagAuditTests(unittest.TestCase):
    def test_semantic_components_keep_different_classes_separate(self):
        target = torch.tensor([[1, 1, 2, 2]])
        bags = semantic_component_bags(target, minimum_area=1)
        self.assertEqual([bag.target_label for bag in bags], [1, 2])

    def test_predicted_component_records_background_majority(self):
        loc = torch.ones((2, 2), dtype=torch.long)
        target = torch.tensor([[0, 0], [0, 2]])
        bags = predicted_localization_bags(loc, target, minimum_area=1)
        self.assertEqual(len(bags), 1)
        self.assertEqual(bags[0].target_label, 0)
        self.assertAlmostEqual(bags[0].purity, 0.75)

    def test_bag_correct_pixel_wrong_and_d_to_i_are_counted(self):
        counts = new_counts()
        update_region_counts(
            counts,
            target_label=2,
            pixel_label=1,
            bag_label=2,
            semantic_pixels=torch.tensor([2, 2, 2]),
            pixel_predictions=torch.tensor([1, 1, 2]),
        )
        report = finalize_counts(counts)
        self.assertEqual(report["bag_correct_pixel_wrong"], 1)
        self.assertEqual(report["d_to_i_regions_bag_recovers"], 1)
        self.assertEqual(report["d_to_i_pixels"], 2)
        self.assertEqual(report["d_to_i_pixels_bag_recovers"], 2)
        self.assertAlmostEqual(report["d_to_i_region_recovery_ceiling"], 1.0)

    def test_bag_harm_is_counted_when_pixel_region_is_correct(self):
        counts = new_counts()
        update_region_counts(
            counts,
            target_label=3,
            pixel_label=3,
            bag_label=2,
            semantic_pixels=torch.tensor([3, 3]),
            pixel_predictions=torch.tensor([3, 3]),
        )
        report = finalize_counts(counts)
        self.assertEqual(report["bag_wrong_pixel_correct"], 1)
        self.assertAlmostEqual(report["bag_harm_given_pixel_correct"], 1.0)

    def test_hierarchical_factorization_reversal_is_separate(self):
        counts = new_counts()
        update_region_counts(
            counts,
            target_label=2,
            pixel_label=1,
            bag_label=1,
            existence_label=1,
            severity_label=0,
            semantic_pixels=torch.tensor([2]),
            pixel_predictions=torch.tensor([1]),
        )
        report = finalize_counts(counts)
        self.assertEqual(report["affected_exist_correct_joint_intact"], 1)
        self.assertAlmostEqual(report["existence_accuracy"], 1.0)
        self.assertAlmostEqual(report["severity_accuracy_given_affected"], 1.0)
        self.assertAlmostEqual(report["factorization_reversal_given_affected"], 1.0)


if __name__ == "__main__":
    unittest.main()
