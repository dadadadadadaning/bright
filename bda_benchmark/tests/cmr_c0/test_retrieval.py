import unittest

import torch

from model.cmr_c0.case_card import (
    CRITICAL_VARIANTS,
    class_logmeanexp,
    directional_topk_score,
    directional_variant_scores,
)


class CMRRetrievalTests(unittest.TestCase):
    def test_direction_is_query_to_memory(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        matching = query.clone()
        unrelated = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        good = directional_topk_score(query, matching, neighbor_k=1)
        bad = directional_topk_score(query, unrelated, neighbor_k=1)
        self.assertGreater(float(good), float(bad))

    def test_neighbor_k_is_clamped_to_memory_size(self):
        query = torch.tensor([[1.0, 0.0]])
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        score = directional_topk_score(query, memory, neighbor_k=9)
        self.assertAlmostEqual(float(score), 0.5, places=6)

    def test_counterfactual_d_minus_i(self):
        query = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
        intact_cases = [
            directional_topk_score(query, torch.tensor([[0.0, 1.0]]), neighbor_k=1),
            directional_topk_score(query, torch.tensor([[-1.0, 0.0]]), neighbor_k=1),
        ]
        damaged_cases = [
            directional_topk_score(query, torch.tensor([[1.0, 0.0]]), neighbor_k=1),
            directional_topk_score(query, torch.tensor([[0.8, 0.2]]), neighbor_k=1),
        ]
        s_i = class_logmeanexp(torch.stack(intact_cases), 0.1)
        s_d = class_logmeanexp(torch.stack(damaged_cases), 0.1)
        self.assertGreater(float(s_d - s_i), 0.0)

    def test_all_variants_reuse_same_pair_without_changing_math(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        indices = {
            variant: torch.tensor([0, 1]) if variant != "all" else torch.arange(3)
            for variant in CRITICAL_VARIANTS
        }
        scores = directional_variant_scores(
            query, memory, indices, indices, neighbor_k=1
        )
        expected = directional_topk_score(
            query[indices["di_top20"]], memory[indices["di_top20"]], neighbor_k=1
        )
        self.assertAlmostEqual(float(scores["di_top20"]), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
