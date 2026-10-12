import unittest

import torch

from bda_benchmark.experiments.spi_s2_evidence_ladder_v1.router import IsolatedEvidenceGate


class RouterTests(unittest.TestCase):
    def test_isolated_gate_refuses_stacking(self):
        with self.assertRaises(ValueError):
            IsolatedEvidenceGate(["cross_event_memory", "within_modality_relation_difference"])

    def test_combination_requires_two_groups(self):
        with self.assertRaises(ValueError):
            IsolatedEvidenceGate(["cross_event_memory"], allow_combination=True)

    def test_output_is_zero_outside_changed_support(self):
        model = IsolatedEvidenceGate(["within_modality_relation_difference"])
        base = torch.randn(1, 35, 3, 3)
        relation = torch.randn(1, 4, 3, 3)
        changed = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        changed[:, :, 1, 1] = True
        output = model(base, {"within_modality_relation_difference": relation}, changed)
        self.assertEqual(int((output > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()

