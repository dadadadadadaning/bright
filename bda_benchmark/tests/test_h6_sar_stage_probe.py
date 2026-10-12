import unittest

import torch

from bda_benchmark.script.standard_ML.audit_H6_SAR_stage_probes import (
    binary_auprc,
    binary_auroc,
    event_name,
    pure_building_cells,
)


class H6SARStageProbeTests(unittest.TestCase):
    def test_event_name_preserves_disaster_suffix(self):
        self.assertEqual(
            event_name("la_palma-volcano_00000123"), "la_palma-volcano"
        )

    def test_binary_metrics_handle_perfect_reverse_and_ties(self):
        labels = [0, 0, 1, 1]
        self.assertEqual(binary_auroc(labels, [0.0, 0.1, 0.9, 1.0]), 1.0)
        self.assertEqual(binary_auroc(labels, [1.0, 0.9, 0.1, 0.0]), 0.0)
        self.assertEqual(binary_auroc(labels, [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertGreater(binary_auprc(labels, [0.0, 0.1, 0.9, 1.0]), 0.9)

    def test_pure_cell_extraction_excludes_background(self):
        feature = torch.randn(1, 5, 2, 2)
        damage = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]
        )
        records = pure_building_cells(
            feature, damage, minimum_purity=0.75, maximum_per_class=4
        )
        labels = torch.cat([record[1] for record in records])
        self.assertEqual(set(labels.tolist()), {1, 2, 3})
        self.assertNotIn(0, labels.tolist())


if __name__ == "__main__":
    unittest.main()
