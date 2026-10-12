import unittest

import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    CalibrationBins,
    complementarity_counts,
    correctness_oracle_prediction,
    make_decision,
    metrics_from_histogram,
    proposal_route_audit,
    transition_records,
)
from bda_benchmark.script.standard_ML.summarize_H7_event_complementarity import (
    disaster_type,
)


class H7P3A2OracleTests(unittest.TestCase):
    def test_complementarity_partition_is_complete(self):
        target = torch.tensor([[0, 1, 2, 3, 255]])
        p3 = torch.tensor([[0, 1, 1, 2, 0]])
        a2 = torch.tensor([[0, 2, 2, 1, 3]])
        counts = complementarity_counts(p3, a2, target)
        self.assertEqual(counts["valid"], 4)
        self.assertEqual(counts["both_correct"], 1)
        self.assertEqual(counts["p3_only_correct"], 1)
        self.assertEqual(counts["a2_only_correct"], 1)
        self.assertEqual(counts["neither_correct"], 1)
        self.assertEqual(
            counts["valid"],
            counts["both_correct"]
            + counts["p3_only_correct"]
            + counts["a2_only_correct"]
            + counts["neither_correct"],
        )

    def test_correctness_oracle_never_loses_a_correct_expert(self):
        target = torch.tensor([[0, 1, 2, 3]])
        p3 = torch.tensor([[0, 1, 1, 2]])
        a2 = torch.tensor([[0, 2, 2, 1]])
        oracle = correctness_oracle_prediction(p3, a2, target)
        self.assertTrue(torch.equal(oracle, torch.tensor([[0, 1, 2, 2]])))
        either_correct = (p3 == target) | (a2 == target)
        self.assertTrue(torch.all(oracle[either_correct] == target[either_correct]))

    def test_metrics_match_identity_confusion(self):
        histogram = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        metrics = metrics_from_histogram(histogram)
        self.assertAlmostEqual(metrics["miou4"], 1.0)
        self.assertAlmostEqual(metrics["valid_miou"], 1.0)
        self.assertAlmostEqual(metrics["damaged_iou"], 1.0)

    def test_calibration_ece_matches_weighted_bin_gap(self):
        bins = CalibrationBins(bin_count=2)
        confidence = torch.tensor([0.25, 0.25, 0.75, 0.75])
        correct = torch.tensor([False, True, True, True])
        valid = torch.ones(4, dtype=torch.bool)
        bins.update(confidence, correct, valid)
        # Bin accuracies are 0.5 and 1.0, so the expected weighted ECE is 0.25.
        self.assertAlmostEqual(bins.result()["ece"], 0.25)

    def test_decision_requires_all_three_oracle_gates(self):
        p3 = {"valid_miou": 0.60, "damaged_iou": 0.40}
        oracle = {"valid_miou": 0.606, "damaged_iou": 0.411}
        events = [
            {"oracle_minus_p3_pp": {"valid_miou": 0.1}},
            {"oracle_minus_p3_pp": {"valid_miou": 0.2}},
        ]
        decision = make_decision(p3, oracle, events, 0.5, 1.0, 0.5)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["recommendation"],
            "COMPLEMENTARITY_CONFIRMED_DESIGN_LEARNED_SELECTOR",
        )

    def test_transition_records_find_damaged_positive_proposal(self):
        cube = torch.zeros(4, 4, 4, dtype=torch.int64)
        cube[2, 1, 2] = 8
        cube[1, 1, 2] = 3
        records = transition_records(cube)
        transition = next(
            row
            for row in records
            if row["p3_class"] == "intact" and row["a2_class"] == "damaged"
        )
        self.assertEqual(transition["net_a2_correct"], 5)
        self.assertAlmostEqual(transition["a2_precision_within_transition"], 8 / 11)

    def test_damaged_route_audit_is_prediction_conditioned(self):
        cube = torch.zeros(4, 4, 4, dtype=torch.int64)
        cube[2, 1, 2] = 8
        cube[1, 1, 2] = 3
        cube[3, 1, 3] = 100
        audit = proposal_route_audit(cube, a2_proposal_class=2)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["a2_correct"], 8)
        self.assertEqual(audit["p3_correct"], 3)
        self.assertEqual(audit["net_a2_correct"], 5)

    def test_disaster_type_is_taken_from_event_suffix(self):
        self.assertEqual(disaster_type("turkey-earthquake"), "earthquake")
        self.assertEqual(disaster_type("la_palma-volcano"), "volcano")
        self.assertEqual(disaster_type("unknown-scene"), "other")


if __name__ == "__main__":
    unittest.main()
