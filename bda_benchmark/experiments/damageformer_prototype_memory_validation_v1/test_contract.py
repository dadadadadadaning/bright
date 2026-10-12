import unittest
import torch
from .validate import apply_one, selected_indices, tiny_cosine_kmeans


class ContractTests(unittest.TestCase):
    def test_fixed_budget(self):
        score = torch.arange(100).float(); candidate = torch.ones(100, dtype=torch.bool)
        self.assertEqual(len(selected_indices(score, candidate, 0.02)), 2)

    def test_intact_only_override(self):
        base = torch.zeros(4, 4, dtype=torch.long); base[2, 1] = 7
        building = base.clone(); out = apply_one(base, building)
        self.assertEqual(int(out[2, 1]), 0); self.assertEqual(int(out[2, 2]), 7)

    def test_kmeans_shape(self):
        points = torch.eye(4)
        self.assertEqual(tuple(tiny_cosine_kmeans(points, 2, 3407).shape), (2, 4))


if __name__ == "__main__": unittest.main()

