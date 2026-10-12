import unittest

import torch

from script.standard_ML.cmr_v1.audit_top20_complement import analyse_split


class ComplementAuditTests(unittest.TestCase):
    def test_full_transitions_and_harm_exposure(self):
        p0 = {
            "labels": torch.tensor([0, 1, 2, 0]),
            "base_predictions": torch.tensor([0, 0, 1, 0]),
            "events": ["a", "b", "c", "d"],
        }
        labels = torch.tensor([
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2],
            [0, 0, 0],
        ])
        candidates = {
            "query_labels": p0["labels"].clone(),
            "candidate_labels": labels,
            "candidate_indices": torch.tensor([
                [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
            ]),
            "candidate_scores": torch.tensor([
                [0.9, 0.8, 0.7], [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7], [0.9, 0.8, 0.7],
            ]),
            "memory_events": torch.arange(12),
        }
        result = analyse_split(p0, candidates)
        self.assertEqual(result["confusion_transitions_3x3"]["damaged_to_intact"]["queries"], 1)
        self.assertEqual(result["confusion_transitions_3x3"]["destroyed_to_damaged"]["queries"], 1)
        self.assertEqual(
            result["harm_exposure"]["all_p3_correct"]["top@3"]["any_wrong_class"], 1
        )
        self.assertEqual(result["missed_damaged_route"]["unused_headroom_at_maxk"], 1)


if __name__ == "__main__":
    unittest.main()
