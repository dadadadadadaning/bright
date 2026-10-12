import unittest

import torch

from bda_benchmark.experiments.spi_core_ring_adapter_v1.model import ParameterMatchedAdapter, adapter_inputs
from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import build_region_maps, scatter_region_mean


class RegionTests(unittest.TestCase):
    def test_four_connected_diagonal_objects_stay_separate(self):
        probability = torch.zeros(1, 1, 4, 4)
        probability[0, 0, 1, 1] = 1
        probability[0, 0, 2, 2] = 1
        maps = build_region_maps(probability, minimum_area=1)
        self.assertEqual(maps.region_count, 2)

    def test_small_components_are_removed_before_pooling(self):
        probability = torch.zeros(1, 1, 5, 5)
        probability[0, 0, 0, 0] = 1
        probability[0, 0, 2:4, 2:4] = 1
        maps = build_region_maps(probability, minimum_area=2)
        self.assertEqual(maps.region_count, 1)
        self.assertEqual(int(maps.labels[0, 0, 0]), 0)

    def test_scatter_pooling_matches_region_mean(self):
        feature = torch.arange(16.0).reshape(1, 1, 4, 4)
        probability = torch.zeros(1, 1, 4, 4)
        probability[:, :, 1:3, 1:3] = 1
        maps = build_region_maps(probability, minimum_area=1)
        table, _ = scatter_region_mean(feature, maps.labels, maps.region_count)
        self.assertAlmostEqual(float(table[1, 0]), float(feature[0, 0, 1:3, 1:3].mean()))

    def test_core_and_ring_do_not_overlap(self):
        probability = torch.zeros(1, 1, 9, 9)
        probability[:, :, 2:7, 2:7] = 1
        maps = build_region_maps(probability, minimum_area=1, core_radius=1, ring_radius=1)
        self.assertFalse(bool(((maps.core_labels > 0) & (maps.ring_labels > 0)).any()))
        self.assertTrue(bool((maps.core_labels > 0).any()))
        self.assertTrue(bool((maps.ring_labels > 0).any()))


class AdapterTests(unittest.TestCase):
    def test_zero_initialization_exactly_preserves_base_logits(self):
        adapter = ParameterMatchedAdapter(8, 8, 4)
        feature = torch.randn(2, 8, 7, 7)
        residual = adapter(feature, feature, torch.zeros_like(feature), torch.ones(2, 1, 7, 7))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_parameter_matched_variants_share_one_architecture(self):
        counts = [sum(p.numel() for p in ParameterMatchedAdapter().parameters()) for _ in range(3)]
        self.assertEqual(len(set(counts)), 1)

    def test_region_adapter_cannot_change_outside_loccc(self):
        feature = torch.randn(1, 256, 7, 7)
        probability = torch.zeros(1, 1, 7, 7)
        probability[:, :, 2:5, 2:5] = 1
        maps = build_region_maps(probability, minimum_area=1)
        context, contrast, support = adapter_inputs("m4_core_ring_region", feature, maps)
        adapter = ParameterMatchedAdapter()
        torch.nn.init.ones_(adapter.out.weight)
        residual = adapter(feature, context, contrast, support)
        outside = (maps.labels == 0).unsqueeze(1).expand_as(residual)
        self.assertTrue(torch.equal(residual[outside], torch.zeros_like(residual[outside])))


if __name__ == "__main__":
    unittest.main()
