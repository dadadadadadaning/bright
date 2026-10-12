import unittest

import torch

from model.cmr_c0.case_card import CRITICAL_VARIANTS
from script.standard_ML.cmr_c0.audit_critical_verifier import build_feature_table


def descriptor(axis: int) -> torch.Tensor:
    value = torch.zeros(2, 256)
    value[:, axis] = 1.0
    return value.half()


class SyntheticAuditPipelineTests(unittest.TestCase):
    def test_fixed_candidates_produce_two_sided_counterfactual_delta(self):
        support_features = [descriptor(0), descriptor(0), descriptor(1), descriptor(1)]
        query_features = [descriptor(1), descriptor(0)]
        support = {
            "memory_local_indices": torch.arange(4),
            "features": support_features,
            "critical_indices": {
                variant: [torch.tensor([0, 1]) for _ in range(4)]
                for variant in CRITICAL_VARIANTS
            },
        }
        query = {
            "labels": torch.tensor([1, 0]),
            "features": query_features,
            "critical_indices": {
                variant: [torch.tensor([0, 1]) for _ in range(2)]
                for variant in CRITICAL_VARIANTS
            },
            "candidate_indices": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            "candidate_labels": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
            "candidate_scores": torch.tensor([[0.2, 0.1, 0.9, 0.8], [0.9, 0.8, 0.2, 0.1]]),
            "base_probabilities": torch.tensor([[0.8, 0.15, 0.05], [0.8, 0.15, 0.05]]),
            "events": ["libya-flood", "turkey-earthquake"],
            "disaster_types": ["flood", "earthquake"],
        }
        table = build_feature_table(
            query, support, device="cpu", neighbor_k=1, class_temperature=0.1
        )
        delta = table["groups"]["di_top20_set_only"]["features"][:, 2]
        self.assertGreater(float(delta[0]), 0.0)
        self.assertLess(float(delta[1]), 0.0)
        self.assertEqual(table["availability"]["retained_rows"], 2)
        self.assertEqual(table["direct"]["di_top20"]["unused_d_headroom"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
