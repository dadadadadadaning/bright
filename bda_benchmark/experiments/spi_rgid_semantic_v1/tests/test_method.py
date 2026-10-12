import unittest

import numpy as np
import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.method import (
    SplitRiskConfig,
    random_matched_partition,
    score_split_risk,
    selective_partition,
)


def _inputs():
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    probabilities = torch.full((4, 12, 12), 0.25)
    center = torch.zeros((12, 12))
    center[4, 4] = 0.95
    center[7, 7] = 0.90
    boundary = torch.ones((12, 12))
    return mask, probabilities, center, boundary


class SplitRiskTests(unittest.TestCase):
    def test_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            SplitRiskConfig(weight_area=0.20)

    def test_one_child_is_never_accepted(self):
        mask, probabilities, center, boundary = _inputs()
        evidence = score_split_risk(
            parent_id=1,
            parent_mask=mask,
            probabilities=probabilities,
            center_probability=center,
            boundary_probability=boundary,
            child_count=1,
            config=SplitRiskConfig(score_threshold=0.0),
        )
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.rejection_reason, "rgid_has_fewer_than_two_children")

    def test_high_risk_two_peak_parent_is_accepted(self):
        mask, probabilities, center, boundary = _inputs()
        evidence = score_split_risk(
            parent_id=7,
            parent_mask=mask,
            probabilities=probabilities,
            center_probability=center,
            boundary_probability=boundary,
            child_count=2,
            config=SplitRiskConfig(score_threshold=0.0),
        )
        self.assertTrue(evidence.accepted)
        self.assertEqual(evidence.peak_count, 2)

    def test_selective_partition_splits_only_accepted_parent(self):
        parents = np.zeros((4, 8), dtype=np.int32)
        parents[:, :4] = 1
        parents[:, 4:] = 2
        children = np.zeros_like(parents)
        children[:2, :4] = 10
        children[2:, :4] = 11
        children[:2, 4:] = 20
        children[2:, 4:] = 21
        mask, probabilities, center, boundary = _inputs()
        accepted = score_split_risk(
            parent_id=1,
            parent_mask=mask,
            probabilities=probabilities,
            center_probability=center,
            boundary_probability=boundary,
            child_count=2,
            config=SplitRiskConfig(score_threshold=0.0),
        )
        rejected = score_split_risk(
            parent_id=2,
            parent_mask=mask,
            probabilities=probabilities,
            center_probability=center,
            boundary_probability=boundary,
            child_count=1,
            config=SplitRiskConfig(score_threshold=0.0),
        )
        output = selective_partition(
            parent_labels=parents,
            rgid_child_labels=children,
            evidence_by_parent={1: accepted, 2: rejected},
        )
        self.assertEqual(len(np.unique(output[parents == 1])), 2)
        self.assertEqual(len(np.unique(output[parents == 2])), 1)
        np.testing.assert_array_equal(output > 0, parents > 0)

    def test_support_mismatch_is_rejected(self):
        parents = np.ones((3, 3), dtype=np.int32)
        children = parents.copy()
        children[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "same support"):
            selective_partition(
                parent_labels=parents,
                rgid_child_labels=children,
                evidence_by_parent={},
            )

    def test_random_control_splits_exactly_the_requested_parent_count(self):
        parents = np.repeat(np.arange(1, 5, dtype=np.int32), 4).reshape(4, 4)
        children = np.zeros_like(parents)
        for parent_id in range(1, 5):
            mask = parents == parent_id
            positions = np.argwhere(mask)
            for index, position in enumerate(positions):
                children[tuple(position)] = 10 * parent_id + index // 2
        output, selected = random_matched_partition(
            parent_labels=parents,
            rgid_child_labels=children,
            split_parent_count=2,
            seed=3407,
        )
        self.assertEqual(len(selected), 2)
        split_count = sum(
            len(np.unique(output[parents == parent_id])) > 1
            for parent_id in range(1, 5)
        )
        self.assertEqual(split_count, 2)
        np.testing.assert_array_equal(output > 0, parents > 0)


if __name__ == "__main__":
    unittest.main()
