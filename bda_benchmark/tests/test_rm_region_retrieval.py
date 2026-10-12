# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import torch

from script.standard_ML.rm_region_retrieval_runtime import (
    balanced_indices,
    binary_auroc,
    build_combined_keys,
    build_raw_only_keys,
    class_event_balanced_indices,
    class_event_stratum_weights,
    class_balanced_knn,
    complement_audit,
    event_from_name,
    gain_retention_audit,
    p3_anchored_memory_fusion,
    prior_corrected_retrieval,
    raw_stat_normalizer,
    scene_relative_raw_stats,
    selective_damaged_rescue_audit,
    apply_dual_key_reject_rule,
    shuffled_labels,
    validate_disjoint_ids,
    weighted_complement_audit,
)


class RMRegionRetrievalTests(unittest.TestCase):
    def test_gain_retention_is_not_infinite_when_ordinary_is_below_p3(self):
        audit = gain_retention_audit(0.80, 0.68, 0.71)
        self.assertIsNone(audit["gain_retention"])
        self.assertEqual(
            audit["check_name"],
            "candidate_beats_p3_when_ordinary_has_no_gain",
        )
        self.assertFalse(audit["passed"])

    def test_binary_auroc_and_selective_rescue_confidence(self):
        self.assertEqual(
            binary_auroc(torch.tensor([0.1, 0.9]), torch.tensor([0, 1])),
            1.0,
        )
        truth = torch.tensor([1, 0, 1, 0])
        base = torch.zeros(4, dtype=torch.long)
        base_probabilities = torch.tensor([
            [0.45, 0.45, 0.10],
            [0.90, 0.08, 0.02],
            [0.48, 0.42, 0.10],
            [0.88, 0.10, 0.02],
        ])
        retrieval = torch.ones(4, dtype=torch.long)
        retrieval_scores = torch.tensor([
            [0.10, 0.90, 0.05],
            [0.40, 0.45, 0.20],
            [0.15, 0.85, 0.10],
            [0.35, 0.40, 0.25],
        ])
        audit = selective_damaged_rescue_audit(
            truth,
            base,
            base_probabilities,
            retrieval,
            retrieval_scores,
            ["a", "b", "c", "d"],
            coverages=(0.5,),
        )
        self.assertEqual(audit["rescue_proposals"], 2)
        self.assertEqual(audit["harm_proposals"], 2)
        self.assertEqual(audit["signal_auroc"]["retrieval_margin"], 1.0)
        self.assertEqual(audit["operating_points"][0]["net_correction"], 2)

    def test_event_parser_removes_only_tile_index(self):
        self.assertEqual(event_from_name("hawaii-wildfire_00123"), "hawaii-wildfire")

    def test_memory_balancing_is_exact_and_deterministic(self):
        labels = torch.tensor([0] * 8 + [1] * 3 + [2] * 5)
        first = balanced_indices(labels, capacity_per_class=4, seed=3407)
        second = balanced_indices(labels, capacity_per_class=4, seed=3407)
        torch.testing.assert_close(first, second)
        self.assertEqual(torch.bincount(labels[first], minlength=3).tolist(), [3, 3, 3])

    def test_zero_raw_weight_is_exact_visual_key(self):
        visual = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        raw = torch.tensor([[10.0, 2.0], [20.0, 4.0]])
        normalizer = raw_stat_normalizer(raw)
        actual = build_combined_keys(visual, raw, normalizer, raw_weight=0.0)
        expected = torch.nn.functional.normalize(visual, dim=1)
        torch.testing.assert_close(actual, expected)

    def test_raw_only_key_is_finite_and_normalised(self):
        raw = torch.tensor([[0.0, 1.0], [2.0, 4.0]])
        normalizer = {
            "sar_mean_center": 1.0,
            "sar_mean_scale": 1.0,
            "sar_logstd_center": 0.0,
            "sar_logstd_scale": 1.0,
        }
        key = build_raw_only_keys(raw, normalizer)
        self.assertTrue(torch.isfinite(key).all())
        torch.testing.assert_close(key.norm(dim=1), torch.ones(2))

    def test_dual_key_reject_requires_every_gate_and_preserves_others(self):
        base = torch.tensor([0, 0, 0, 2])
        signals = {
            "visual_damaged_vs_intact_llr": torch.tensor([2.0, 2.0, 0.1, 3.0]),
            "sar_damaged_vs_intact_llr": torch.tensor([2.0, 0.1, 2.0, 3.0]),
            "p3_uncertainty": torch.tensor([0.9, 0.9, 0.9, 0.9]),
            "damaged_support": torch.tensor([0.8, 0.8, 0.8, 0.8]),
            "damaged_event_consensus": torch.tensor([3.0, 3.0, 3.0, 3.0]),
        }
        rule = {
            "enabled": True,
            "visual_llr_threshold": 1.0,
            "sar_llr_threshold": 1.0,
            "p3_uncertainty_threshold": 0.5,
            "damaged_support_threshold": 0.5,
            "minimum_damaged_events": 2.0,
        }
        result = apply_dual_key_reject_rule(base, signals, rule)
        torch.testing.assert_close(result["prediction"], torch.tensor([1, 0, 0, 2]))
        rejected = apply_dual_key_reject_rule(base, signals, {"enabled": False})
        torch.testing.assert_close(rejected["prediction"], base)

    def test_class_event_memory_is_class_balanced_and_event_capped(self):
        labels = torch.tensor([0] * 12 + [1] * 12 + [2] * 12)
        events = (["a"] * 8 + ["b"] * 4) * 3
        selected = class_event_balanced_indices(
            labels, events, capacity_per_class=6,
            capacity_per_event_class=3, seed=3407,
        )
        self.assertEqual(torch.bincount(labels[selected], minlength=3).tolist(), [6, 6, 6])
        for cls in range(3):
            for event in ("a", "b"):
                count = sum(
                    int(labels[index]) == cls and events[index] == event
                    for index in selected.tolist()
                )
                self.assertLessEqual(count, 3)

    def test_scene_relative_stats_remove_affine_scene_shift(self):
        region = torch.tensor([[12.0, 4.0], [30.0, 6.0]])
        scene = torch.tensor([[10.0, 2.0], [20.0, 5.0]])
        reference = scene_relative_raw_stats(region, scene)
        transformed = scene_relative_raw_stats(3.0 * region + torch.tensor([7.0, 0.0]), 3.0 * scene + torch.tensor([7.0, 0.0]))
        torch.testing.assert_close(reference, transformed)

    def test_class_balanced_knn_scores_each_class(self):
        memory = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 2])
        query = torch.tensor([[0.0, 1.0]])
        result = class_balanced_knn(query, memory, labels, topk_per_class=1)
        self.assertEqual(int(result["prediction"][0]), 1)

    def test_zero_prior_strength_exactly_preserves_retrieval_decision(self):
        scores = torch.tensor([[0.8, 0.7, 0.1], [0.2, 0.4, 0.9]])
        corrected = prior_corrected_retrieval(scores, [80, 10, 10], tau=0.0)
        torch.testing.assert_close(corrected["prediction"], scores.argmax(dim=1))

    def test_train_prior_correction_can_reject_rare_class_overprediction(self):
        scores = torch.tensor([[0.50, 0.55, 0.10]])
        corrected = prior_corrected_retrieval(
            scores, [840, 80, 80], tau=1.0, smoothing=0.0
        )
        self.assertEqual(int(corrected["prediction"][0]), 0)

    def test_p3_anchored_memory_alpha_zero_is_exact(self):
        base = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
        memory = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
        fused = p3_anchored_memory_fusion(base, memory, alpha=0.0)
        torch.testing.assert_close(fused["probabilities"], base)
        torch.testing.assert_close(fused["prediction"], base.argmax(dim=1))

    def test_p3_anchored_memory_can_modify_an_uncertain_decision(self):
        base = torch.tensor([[0.51, 0.48, 0.01]])
        memory = torch.tensor([[0.10, 0.90, 0.01]])
        fused = p3_anchored_memory_fusion(base, memory, alpha=0.5)
        self.assertEqual(int(fused["prediction"][0]), 1)

    def test_class_event_weights_reconstruct_prebalance_mass(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        events = ["a", "b", "a", "b", "a", "b"]
        before = {
            "intact": {"a": 100, "b": 200},
            "damaged": {"a": 10, "b": 20},
            "destroyed": {"a": 5, "b": 15},
        }
        after = {
            "intact": {"a": 1, "b": 1},
            "damaged": {"a": 1, "b": 1},
            "destroyed": {"a": 1, "b": 1},
        }
        weights = class_event_stratum_weights(labels, events, before, after)
        self.assertEqual(float(weights.sum()), 350.0)
        audit = weighted_complement_audit(
            labels,
            torch.tensor([0, 0, 0, 0, 2, 2]),
            torch.tensor([0, 1, 1, 1, 2, 2]),
            weights,
            events,
        )
        self.assertEqual(audit["per_class"]["damaged"]["net"], 30.0)

    def test_same_event_neighbours_are_excluded(self):
        memory = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0], [-0.8, 0.2], [0.0, -1.0]])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        result = class_balanced_knn(
            torch.tensor([[1.0, 0.0]]), memory, labels,
            topk_per_class=1,
            query_events=["event-a"],
            memory_events=["event-a", "event-b", "event-b", "event-a", "event-b", "event-a"],
            exclude_same_event=True,
        )
        self.assertTrue(torch.isfinite(result["scores"]).all())

    def test_event_diverse_knn_prevents_one_event_from_filling_topk(self):
        memory = torch.tensor([
            [1.0, 0.0], [0.99, 0.10], [0.0, 1.0],
            [0.8, 0.6], [0.8, 0.6], [-1.0, 0.0], [-0.9, 0.1],
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        events = ["a", "a", "b", "a", "b", "a", "b"]
        standard = class_balanced_knn(
            torch.tensor([[1.0, 0.0]]), memory, labels, topk_per_class=2,
            query_events=["q"], memory_events=events,
        )
        diverse = class_balanced_knn(
            torch.tensor([[1.0, 0.0]]), memory, labels, topk_per_class=2,
            query_events=["q"], memory_events=events, max_neighbors_per_event=1,
        )
        # The contract is about neighbour composition, not about forcing an
        # artificial final-class flip.  Standard top-2 for class 0 both come
        # from event "a"; the capped version must use events "a" and "b".
        self.assertEqual(
            float(standard["unique_neighbor_events_by_class"][0, 0]), 1.0
        )
        self.assertEqual(
            float(diverse["unique_neighbor_events_by_class"][0, 0]), 2.0
        )
        self.assertLess(
            float(diverse["scores"][0, 0]), float(standard["scores"][0, 0])
        )

    def test_shuffle_preserves_class_counts(self):
        labels = torch.tensor([0, 0, 1, 2, 2, 2])
        shuffled = shuffled_labels(labels, seed=7)
        self.assertEqual(torch.bincount(shuffled, minlength=3).tolist(), [2, 1, 3])

    def test_complement_counts_rescue_minus_harm(self):
        truth = torch.tensor([1, 1, 2, 0])
        base = torch.tensor([0, 1, 1, 0])
        retrieval = torch.tensor([1, 0, 2, 0])
        audit = complement_audit(truth, base, retrieval)
        self.assertEqual(audit["net_complement"], 1)
        self.assertEqual(audit["per_class"]["damaged"]["net_complement"], 0)

    def test_split_overlap_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_disjoint_ids(["a", "b"], ["b", "c"])


if __name__ == "__main__":
    unittest.main()
