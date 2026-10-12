import unittest
import torch
from .cohorts import difficult_retrieval


class DifficultRetrievalContracts(unittest.TestCase):
    def test_fixed_cohort_and_unsupported_mass_not_faked(self):
        bank = {'distribution': torch.eye(4)}
        q = torch.ones(1, 1, 2)
        ids = torch.tensor([[0, 1]])
        selected = torch.tensor([[2, 1], [-1, -1]])
        target = torch.tensor([[[2, 2]]])
        reference = torch.tensor([[[1, 0]]])
        row = difficult_retrieval(target, reference, selected, bank, q, ids, (1, 2))
        d = row['damaged_j1_wrong']
        self.assertEqual(d['count'], 2)
        self.assertEqual(d['support_mass_sum'], 1)
        self.assertEqual(d['damaged_hit_mass_sum'], 1)
        self.assertEqual(d['damaged_neighbor_fraction_mass_sum'], .5)
        self.assertEqual(d['intact_neighbor_fraction_mass_sum'], .5)
        self.assertEqual(row['damaged_j1_correct']['count'], 0)

    def test_q_mixing_retains_fractional_evidence(self):
        bank = {'distribution': torch.eye(4)}
        q = torch.tensor([[[.25], [.75]]])
        ids = torch.tensor([[0], [1]])
        selected = torch.tensor([[2], [1]])
        row = difficult_retrieval(torch.tensor([[[2]]]), torch.tensor([[[1]]]),
                                 selected, bank, q, ids, (1, 1))['damaged_j1_wrong']
        self.assertEqual(row['damaged_hit_mass_sum'], .25)
        self.assertEqual(row['intact_neighbor_fraction_mass_sum'], .75)


if __name__ == '__main__':
    unittest.main()
