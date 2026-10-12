import unittest

import torch

from script.standard_ML.hm_ob0_boundary_runtime import (
    boundary_candidate_table,
    operating_point_passes,
    risk_coverage_table,
    select_train_operating_point,
    strict_event_crossfit_scores,
)


class HMOB0BoundaryTests(unittest.TestCase):
    def _cache(self):
        return {
            "labels": torch.tensor([0, 1, 2, 0]),
            "base_predictions": torch.tensor([0, 0, 1, 0]),
            "base_probabilities": torch.tensor([
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
            ]),
            "events": ["a", "a", "b", "b"],
            "key_variants": {"mean": torch.eye(4)},
        }

    def test_candidate_pool_contains_only_p3_intact_rows(self):
        table = boundary_candidate_table(self._cache())
        self.assertEqual(table["indices"].tolist(), [0, 1, 3])
        self.assertEqual(table["truth"].tolist(), [0, 1, 0])
        self.assertEqual(table["missed_damaged"], 1)
        self.assertEqual(table["missed_destroyed"], 0)
        self.assertEqual(table["true_intact"], 2)

    def test_risk_coverage_counts_rescue_and_harm(self):
        rows = risk_coverage_table(
            torch.tensor([0.9, 0.8, 0.1, 0.0]),
            torch.tensor([1, 2, 0, 0]),
            ["a", "b", "a", "b"],
            coverages=(0.5,),
        )
        row = rows[0]
        self.assertEqual(row["rescued"], 2)
        self.assertEqual(row["damaged_rescued"], 1)
        self.assertEqual(row["destroyed_rescued"], 1)
        self.assertEqual(row["harmed"], 0)
        self.assertEqual(row["net_correction"], 2)

    def test_locked_point_requires_every_predeclared_safety_check(self):
        good = {
            "coverage": 0.1,
            "score_threshold": 0.8,
            "precision": 0.6,
            "net_correction": 2,
            "rescued": 12,
            "nonnegative_events": 10,
            "positive_events": 3,
        }
        kwargs = {
            "minimum_precision": 0.55,
            "minimum_rescued": 10,
            "minimum_nonnegative_events": 10,
            "minimum_positive_events": 3,
        }
        self.assertIsNotNone(select_train_operating_point([good], **kwargs))
        self.assertTrue(operating_point_passes(good, **kwargs))
        bad = dict(good, positive_events=2)
        self.assertIsNone(select_train_operating_point([bad], **kwargs))
        self.assertFalse(operating_point_passes(bad, **kwargs))

    def test_strict_crossfit_scores_every_candidate_event(self):
        table = {
            "mean_features": torch.tensor([
                [1.0, 0.0], [0.0, 1.0],
                [0.9, 0.1], [0.1, 0.9],
                [0.8, 0.2], [0.2, 0.8],
            ]),
            "all_occurrence_labels": torch.tensor([0, 1, 0, 1, 0, 1]),
            "indices": torch.arange(6),
            "occurrence_target": torch.tensor([0, 1, 0, 1, 0, 1]).bool(),
            "p3_affected_probability": torch.tensor(
                [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
            ),
        }
        result = strict_event_crossfit_scores(
            table,
            torch.tensor([0, 0, 1, 1, 2, 2]),
            cap_per_event_class=10,
            seed=3407,
            logistic_l2=1e-2,
            logistic_max_iter=50,
        )
        self.assertEqual(len(result["folds"]), 3)
        for scores in result["scores"].values():
            self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
