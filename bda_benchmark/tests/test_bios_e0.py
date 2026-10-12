import unittest

import torch

from bda_benchmark.script.standard_ML.bios_e0_runtime import (
    aggregate_token_scores,
    binary_auprc,
    binary_auroc,
    complete_signal_metrics,
    deterministic_valid_shuffle,
    relation_token_anomaly,
    shift_token_grid,
    spearman_correlation,
)


class BIOSE0Tests(unittest.TestCase):
    def test_identical_modalities_have_zero_relation_anomaly(self):
        torch.manual_seed(7)
        tokens = torch.randn(9, 5)
        valid = torch.ones(9, dtype=torch.bool)
        for spatial in ("local", "global", "local_global"):
            values = relation_token_anomaly(
                tokens, tokens.clone(), valid,
                grid_size=3, spatial=spatial, relation="sim",
            )
            self.assertTrue(torch.allclose(values, torch.zeros_like(values), atol=1e-6))

    def test_relation_anomaly_detects_a_structural_change(self):
        tokens = torch.eye(9)
        changed = tokens.clone()
        changed[4] = changed[0]
        valid = torch.ones(9, dtype=torch.bool)
        values = relation_token_anomaly(
            tokens, changed, valid,
            grid_size=3, spatial="local_global", relation="sim",
        )
        self.assertGreater(float(values.max()), 0.0)
        self.assertGreater(float(values[4]), float(values.mean()))

    def test_similarity_dissimilarity_mode_is_finite(self):
        torch.manual_seed(11)
        rgb = torch.randn(16, 8)
        sar = torch.randn(16, 8)
        valid = torch.ones(16, dtype=torch.bool)
        values = relation_token_anomaly(
            rgb, sar, valid, grid_size=4, spatial="global",
            relation="sd", neighbour_count=3,
        )
        self.assertEqual(tuple(values.shape), (16,))
        self.assertTrue(torch.isfinite(values).all())

    def test_shift_has_no_circular_wraparound(self):
        tokens = torch.arange(9, dtype=torch.float32).reshape(9, 1)
        valid = torch.ones(9, dtype=torch.bool)
        shifted, shifted_valid = shift_token_grid(tokens, valid, 3, 0, 1)
        grid = shifted.reshape(3, 3)
        mask = shifted_valid.reshape(3, 3)
        self.assertTrue(torch.equal(grid[:, 1:], torch.tensor([[0., 1.], [3., 4.], [6., 7.]])))
        self.assertTrue(torch.equal(mask[:, 0], torch.zeros(3, dtype=torch.bool)))
        self.assertFalse(bool((grid[:, 0] == torch.tensor([2., 5., 8.])).all()))

    def test_shuffle_is_deterministic_and_only_moves_valid_tokens(self):
        tokens = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        valid = torch.tensor([True, True, False, True, False, True])
        first = deterministic_valid_shuffle(tokens, valid, 3407)
        second = deterministic_valid_shuffle(tokens, valid, 3407)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[~valid], tokens[~valid]))
        self.assertCountEqual(first[valid].flatten().tolist(), tokens[valid].flatten().tolist())

    def test_binary_metrics_and_ordinal_correlation_are_exact(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(binary_auroc(labels, scores), 1.0)
        self.assertAlmostEqual(binary_auprc(labels, scores), 1.0)
        self.assertAlmostEqual(spearman_correlation([0, 1, 2], [0.1, 0.5, 0.9]), 1.0)

    def test_complete_metrics_measure_p3_nonaffected_damaged_misses(self):
        rows = [
            {"label": 1, "p3_prediction": 1, "event": "a", "score": 0.1},
            {"label": 2, "p3_prediction": 1, "event": "a", "score": 0.9},
            {"label": 1, "p3_prediction": 0, "event": "b", "score": 0.2},
            {"label": 2, "p3_prediction": 0, "event": "b", "score": 0.8},
            {"label": 3, "p3_prediction": 3, "event": "b", "score": 1.0},
        ]
        metrics = complete_signal_metrics(rows, "score")
        miss = metrics["p3_nonaffected_miss"]
        self.assertAlmostEqual(miss["auroc"], 1.0)
        self.assertAlmostEqual(miss["event_macro_auroc"], 1.0)
        self.assertEqual(miss["nonnegative_events"], 2)
        self.assertEqual(miss["events_above_0p55"], 2)
        self.assertEqual(miss["events_above_0p60"], 2)
        self.assertEqual(metrics["affected"]["positive"], 3)
        self.assertAlmostEqual(metrics["class_medians"]["damaged"], 0.85)

    def test_aggregates_include_sparse_and_extent_statistics(self):
        values = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0])
        result = aggregate_token_scores(values, threshold=0.5)
        self.assertEqual(set(result), {
            "mean", "std", "max", "q75", "q90", "top20", "area", "concentration"
        })
        self.assertAlmostEqual(result["max"], 1.0)
        self.assertAlmostEqual(result["top20"], 1.0)
        self.assertAlmostEqual(result["area"], 0.4)
        self.assertTrue(torch.isnan(torch.tensor(result["concentration"])))

    def test_concentration_reports_connected_anomaly_fraction(self):
        valid = torch.ones(9, dtype=torch.bool)
        values = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        result = aggregate_token_scores(
            values, threshold=0.5, valid=valid, grid_size=3
        )
        self.assertAlmostEqual(result["area"], 1 / 3)
        self.assertAlmostEqual(result["concentration"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
