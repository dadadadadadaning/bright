import unittest

import numpy as np

from bda_benchmark.experiments.p5_ablation_validation_v1.metrics import (
    aggregate_by_event,
    compare_events,
    event_from_name,
    metrics_from_confusion,
    paired_bootstrap_gain,
)


class MetricTest(unittest.TestCase):
    def test_event_name(self):
        self.assertEqual(event_from_name("hurricane-harvey_000001"), "hurricane-harvey")

    def test_perfect_metrics(self):
        metrics = metrics_from_confusion(np.diag([5, 6, 7, 8]))
        self.assertEqual(metrics["foreground_miou"], 1.0)
        self.assertEqual(metrics["damaged_f1"], 1.0)
        self.assertEqual(metrics["damaged_to_intact"], 0.0)

    def test_event_and_paired_bootstrap(self):
        reference = [
            {"name": "a_1", "event": "a", "confusion": np.diag([1, 2, 2, 2]).tolist()},
            {"name": "b_1", "event": "b", "confusion": np.diag([1, 2, 2, 2]).tolist()},
        ]
        candidate = [dict(row) for row in reference]
        event_ref = aggregate_by_event(reference)
        event_cand = aggregate_by_event(candidate)
        comparison = compare_events(event_cand, event_ref)
        self.assertEqual(comparison["nonnegative_events"], 2)
        bootstrap = paired_bootstrap_gain(candidate, reference, repeats=20)
        self.assertAlmostEqual(bootstrap["observed_gain_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
