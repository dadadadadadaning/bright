import unittest

import numpy as np

from bda_benchmark.experiments.spi_rgid_semantic_v1.fast_targets import (
    build_center_target_fast,
)
from bda_benchmark.experiments.rgid_s9ic_clean.rgid.config import RGIDConfig
from bda_benchmark.experiments.rgid_s9ic_clean.rgid.targets_losses import (
    build_center_target,
)


class FastTargetEquivalenceTests(unittest.TestCase):
    def test_fast_center_is_equivalent_to_legacy_center(self):
        labels = np.zeros((96, 112), dtype=np.int32)
        labels[7:25, 9:31] = 1
        labels[40:80, 50:58] = 2
        labels[45:67, 70:98] = 3
        labels[55:60, 80:86] = 0
        config = RGIDConfig()
        expected, expected_stats = build_center_target(labels, (24, 28), config)
        observed, observed_stats = build_center_target_fast(labels, (24, 28), config)
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(observed_stats.valid_instances, expected_stats.valid_instances)
        self.assertEqual(
            observed_stats.ignored_border_instances,
            expected_stats.ignored_border_instances,
        )
        self.assertEqual(observed_stats.center_collisions, expected_stats.center_collisions)

    def test_border_instance_is_ignored_identically(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:8, 5:12] = 1
        config = RGIDConfig()
        expected, expected_stats = build_center_target(labels, (8, 8), config)
        observed, observed_stats = build_center_target_fast(labels, (8, 8), config)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(observed_stats.ignored_border_instances, 1)
        self.assertEqual(expected_stats.ignored_border_instances, 1)


if __name__ == "__main__":
    unittest.main()
