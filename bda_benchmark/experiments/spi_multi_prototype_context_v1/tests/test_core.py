import unittest

import torch

from bda_benchmark.experiments.spi_multi_prototype_context_v1.core import MultiPrototypeConfig, multi_prototype_context


class MultiPrototypeTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(seed_confidence=0.70, minimum_seed_pixels=1, feature_temperature=0.10,
                      maximum_alpha=0.75, uncertainty_threshold=0.40, assignment_margin=0.0)
        values.update(changes)
        return MultiPrototypeConfig(**values)

    @staticmethod
    def two_class_case():
        probabilities = torch.tensor([[[[0.05, 0.05, 0.05]],
                                       [[0.90, 0.45, 0.05]],
                                       [[0.03, 0.45, 0.88]],
                                       [[0.02, 0.05, 0.02]]]])
        probabilities /= probabilities.sum(1, keepdim=True)
        feature = torch.tensor([[[[1.0, 0.9, 0.0]], [[0.0, 0.1, 1.0]]]])
        labels = torch.ones((1, 1, 3), dtype=torch.long)
        return probabilities, feature, labels

    def test_two_confident_classes_create_eligible_region(self):
        probabilities, feature, labels = self.two_class_case()
        _, _, _, record = multi_prototype_context(probabilities, feature, labels,
                                                config=self.config(), use_feature_assignment=True)
        self.assertEqual(record["eligible_regions"], 1)
        self.assertEqual(record["available_prototypes"], 2)

    def test_single_prototype_region_remains_exact(self):
        probabilities, feature, labels = self.two_class_case()
        probabilities[:, 1] = 0.90
        probabilities[:, 2] = 0.03
        probabilities[:, 3] = 0.02
        probabilities[:, 0] = 0.05
        refined, eligible, selective, record = multi_prototype_context(
            probabilities, feature, labels, config=self.config(), use_feature_assignment=True
        )
        self.assertTrue(torch.equal(refined, probabilities))
        self.assertFalse(bool(eligible.any()))
        self.assertFalse(bool(selective.any()))
        self.assertEqual(record["eligible_regions"], 0)

    def test_background_outside_region_is_unchanged(self):
        probabilities, feature, labels = self.two_class_case()
        labels[0, 0, 1] = 0
        refined, _, _, _ = multi_prototype_context(
            probabilities, feature, labels, config=self.config(), use_feature_assignment=True
        )
        self.assertTrue(torch.equal(refined[:, :, 0, 1], probabilities[:, :, 0, 1]))

    def test_separate_regions_do_not_share_prototypes(self):
        probabilities, feature, labels = self.two_class_case()
        labels = torch.tensor([[[1, 0, 2]]])
        refined, _, _, record = multi_prototype_context(
            probabilities, feature, labels, config=self.config(), use_feature_assignment=True
        )
        self.assertTrue(torch.equal(refined, probabilities))
        self.assertEqual(record["eligible_regions"], 0)

    def test_feature_assignment_is_distinct_from_probability_assignment(self):
        probabilities, feature, labels = self.two_class_case()
        probability_refined, _, _, _ = multi_prototype_context(
            probabilities, feature, labels, config=self.config(), use_feature_assignment=False
        )
        feature_refined, _, _, _ = multi_prototype_context(
            probabilities, feature, labels, config=self.config(), use_feature_assignment=True
        )
        self.assertFalse(torch.allclose(probability_refined, feature_refined))


if __name__ == "__main__":
    unittest.main()
