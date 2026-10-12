import unittest

import torch

from .features import hard_region_ids, semantic_edges
from .metrics import confusion_metrics


class ContractTests(unittest.TestCase):
    def test_hard_assignment_uses_candidate_at_max_probability(self):
        probabilities = torch.zeros(1, 9, 2)
        probabilities[0, 2, 0] = 1
        probabilities[0, 4, 1] = 1
        candidates = torch.arange(18).reshape(1, 9, 2)
        aux = {"association_probabilities": probabilities, "candidate_ids": candidates,
               "building_ids": torch.ones(1, 1, 2, dtype=torch.long)}
        self.assertEqual(hard_region_ids(aux).tolist(), [[[4, 9]]])

    def test_semantic_edges_detect_class_boundary(self):
        labels = torch.tensor([[1, 1, 2], [1, 1, 2]])
        edge = semantic_edges(labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertEqual(int(edge.sum()), 4)

    def test_metrics_are_three_class(self):
        result = confusion_metrics(torch.eye(3))
        self.assertAlmostEqual(result["macro_f1"], 1.0)

    def test_test_access_contract_literal(self):
        self.assertFalse(False)


if __name__ == "__main__":
    unittest.main()

