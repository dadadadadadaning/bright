# -*- coding: utf-8 -*-
import unittest

import torch

from script.standard_ML.im1_matching_reliability_runtime import (
    apply_logistic_calibrator,
    cross_fitted_logistic_scores,
    fit_logistic_calibrator,
    operating_point_at_threshold,
    proposal_signal_table,
    risk_coverage_table,
    select_train_operating_point,
)


class IM1MatchingReliabilityTests(unittest.TestCase):
    def _cache(self):
        return {
            "labels": torch.tensor([1, 0, 2, 1, 0]),
            "base_predictions": torch.tensor([0, 0, 0, 1, 0]),
            "base_probabilities": torch.tensor([
                [0.45, 0.40, 0.15],
                [0.80, 0.15, 0.05],
                [0.40, 0.20, 0.40],
                [0.10, 0.80, 0.10],
                [0.70, 0.20, 0.10],
            ]),
            "events": ["a", "b", "c", "a", "b"],
            "colocated_stats": torch.arange(15, dtype=torch.float32).reshape(5, 3),
            "matching_stats": torch.arange(50, dtype=torch.float32).reshape(5, 10),
            "metadata": {
                "colocated_stat_names": [
                    "cosine_distance_mean", "cosine_distance_std", "cosine_distance_q90"
                ],
                "matching_stat_names": [
                    "one_minus_peak_mean", "one_minus_peak_std", "one_minus_peak_q90",
                    "peak_gain_mean", "peak_gain_q90", "hard_residual_mean",
                    "hard_residual_q90", "margin_mean", "entropy_mean",
                    "confidence_weighted_residual_mean",
                ],
            },
        }

    def test_proposal_contract_and_feature_dimensions(self):
        cache = self._cache()
        prediction = torch.tensor([1, 1, 1, 1, 0])
        scores = torch.tensor([
            [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3],
            [0.1, 0.8, 0.1], [0.8, 0.1, 0.1],
        ])
        table = proposal_signal_table(cache, prediction, scores)
        self.assertEqual(table["auditable_proposals"], 2)
        self.assertEqual(table["rescue_proposals"], 1)
        self.assertEqual(table["harm_proposals"], 1)
        self.assertEqual(table["destroyed_proposals_excluded"], 1)
        self.assertEqual(table["groups"]["im1_a_p3_memory"]["features"].shape[1], 4)
        self.assertEqual(table["groups"]["im1_b_matching"]["features"].shape[1], 7)
        self.assertEqual(table["groups"]["im1_c_combined"]["features"].shape[1], 10)

    def test_logistic_calibrator_learns_a_separable_signal(self):
        features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        target = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
        model = fit_logistic_calibrator(features, target, l2=1e-3)
        score = apply_logistic_calibrator(features, model)
        self.assertGreater(float(score[2]), float(score[1]))
        self.assertGreater(float(score[3]), float(score[0]))

    def test_operating_point_is_selected_only_when_predeclared_checks_pass(self):
        target = torch.tensor([1] * 12 + [0] * 8, dtype=torch.bool)
        scores = torch.linspace(1.0, 0.0, 20)
        events = [f"event-{index % 10}" for index in range(20)]
        rows = risk_coverage_table(scores, target, events, coverages=(0.5, 1.0))
        selected = select_train_operating_point(rows)
        self.assertIsNotNone(selected)
        self.assertEqual(float(selected["coverage"]), 0.5)
        locked = operating_point_at_threshold(
            scores, target, events,
            threshold=selected["score_threshold"],
        )
        self.assertEqual(locked["selected"], selected["selected"])

    def test_event_crossfit_scores_every_item_out_of_fold(self):
        features = torch.tensor([
            [-2.0], [-1.0], [1.0], [2.0],
            [-2.2], [-1.2], [1.2], [2.2],
            [-1.8], [-0.8], [0.8], [1.8],
        ])
        target = torch.tensor([0, 0, 1, 1] * 3, dtype=torch.bool)
        events = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        result = cross_fitted_logistic_scores(
            features, target, events,
            l2=1e-3,
            balance_classes=False,
        )
        scores = result["scores"]
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(len(result["folds"]), 3)
        self.assertGreater(float(scores[target].mean()), float(scores[~target].mean()))


if __name__ == "__main__":
    unittest.main()
