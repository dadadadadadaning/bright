"""Hand-computed retrieval contracts; no dataset, checkpoint or CUDA needed."""

import inspect
import unittest

import torch

from .core import (label_evidence, matched_random_ids,
                   paired_retrieval_diagnostics, select_neighbors,
                   shuffled_label_bank, validate_paired_bank)


def bank_from_2d(vectors, labels=None, events=None, images=None):
    values = torch.tensor(vectors, dtype=torch.float32).reshape(-1, 2)
    rows = values.shape[0]
    t128 = torch.zeros(rows, 128)
    f256 = torch.zeros(rows, 256)
    t128[:, :2], f256[:, :2] = values, values
    if labels is None:
        labels = torch.nn.functional.one_hot(torch.arange(rows) % 4, 4).float()
    else:
        labels = torch.tensor(labels, dtype=torch.float32)
    events = torch.tensor(events if events is not None else [0] * rows)
    images = torch.tensor(images if images is not None else list(range(rows)))
    return {"t128": t128, "f256": f256, "distribution": labels,
            "events": events.long(), "images": images.long(),
            "regions": torch.arange(rows),
            "event_names": [f"event_{index}" for index in range(
                int(events.max()) + 1 if rows else 0)]}


def query2d(values, key="t128"):
    value = torch.tensor(values, dtype=torch.float32).reshape(-1, 2)
    query = torch.zeros(value.shape[0], 128 if key == "t128" else 256)
    query[:, :2] = value
    return query


class PairedRetrievalTests(unittest.TestCase):
    def test_exact_cosine_rank_and_stable_tie(self):
        bank = bank_from_2d([[1, 0], [3, 0], [3, 4], [-1, 0], [0, 2]])
        ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99,
                                      topk=5)
        self.assertEqual(ids.tolist(), [[0, 1, 2, 4, 3]])
        torch.testing.assert_close(scores, torch.tensor([[1., 1., .6, 0., -1.]]))

    def test_strict_event_and_image_exclusion_and_per_query_ids(self):
        bank = bank_from_2d([[1, 0], [3, 4], [0, 1], [-1, 0]],
                            events=[0, 1, 1, 2], images=[10, 11, 12, 13])
        ids, _ = select_neighbors(query2d([[1, 0], [1, 0]]), bank, "t128",
                                  torch.tensor([0, 1]), torch.tensor([11, 13]),
                                  topk=3, chunk_size=1)
        self.assertEqual(ids.tolist(), [[2, 3, -1], [0, -1, -1]])

    def test_same_members_both_key_validity_and_invalid_queries(self):
        bank = bank_from_2d([[1, 0], [0, 1], [-1, 0], [3, 4]])
        bank["f256"][0] = 0
        bank["t128"][1] = 0
        for key in ("t128", "f256"):
            query = query2d([[1, 0], [0, 0]], key)
            ids, scores = select_neighbors(query, bank, key, 9, 99, topk=3)
            self.assertEqual(ids.tolist(), [[3, 2, -1], [-1, -1, -1]])
            self.assertTrue(torch.isneginf(scores[1:]).all())

    def test_per_class_fixed_quota_and_missing_class_not_backfilled(self):
        bank = bank_from_2d([[1, 0], [3, 4], [0, 1], [-1, 0], [0, -1]],
                            labels=[[1, 0, 0, 0], [1, 0, 0, 0],
                                    [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
        ids, _ = select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99,
                                  topk=8, policy="per_class")
        self.assertEqual(ids.tolist(), [[0, 1, 2, -1, 4, 3, -1, -1]])
        with self.assertRaisesRegex(ValueError, "divisible"):
            select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99,
                             topk=7, policy="per_class")

    def test_paired_space_comparison_keeps_values_and_changes_ranking(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        bank["f256"][:, :2] = torch.tensor([[0., 1.], [1., 0.]])
        t_ids, _ = select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99, topk=1)
        f_ids, _ = select_neighbors(query2d([[1, 0]], "f256"), bank, "f256", 9, 99, topk=1)
        self.assertEqual(t_ids.tolist(), [[0]])
        self.assertEqual(f_ids.tolist(), [[1]])

    def test_missing_and_empty_bank_are_safe(self):
        for bank in (bank_from_2d([[1, 0]], events=[0]), bank_from_2d([])):
            ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", 0, 99)
            evidence = label_evidence(bank, ids, scores)
            self.assertEqual(ids.tolist(), [[-1] * 8])
            self.assertFalse(evidence["supported"].item())
            self.assertEqual(evidence["posterior"].tolist(), [[0.] * 4])
            self.assertEqual(evidence["quality"].tolist(), [[0.] * 6])
            self.assertTrue(torch.isfinite(evidence["quality"]).all())

    def test_structure_and_labels_reject_corruption(self):
        bank = bank_from_2d([[1, 0]])
        bad = dict(bank)
        bad["f256"] = torch.zeros(1, 255)
        with self.assertRaisesRegex(ValueError, "f256"):
            validate_paired_bank(bad)
        bad = dict(bank, t128=bank["t128"].clone())
        bad["t128"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_paired_bank(bad)
        for labels in ([[float("nan"), 0, 0, 1]], [[-.1, .1, 0, 1]],
                       [[0, 0, 0, .5]]):
            bad = dict(bank, distribution=torch.tensor(labels))
            with self.assertRaises(ValueError):
                validate_paired_bank(bad)
        with self.assertRaises(ValueError):
            select_neighbors(torch.zeros(1, 127), bank, "t128", 9, 99)
        with self.assertRaisesRegex(ValueError, "sentinel"):
            select_neighbors(query2d([[1, 0]]), bank, "t128", -2, 99)
        invalid_query = query2d([[1, 0]])
        invalid_query[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            select_neighbors(invalid_query, bank, "t128", 9, 99)
        ids, _ = select_neighbors(query2d([[1, 0]]), bank, "t128", -1, -1, topk=1)
        self.assertEqual(ids.tolist(), [[0]])
        with self.assertRaises(ValueError):
            select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99, chunk_size=0)
        with self.assertRaises(ValueError):
            label_evidence(bank, torch.tensor([[1]]), torch.tensor([[.5]]))

    def test_query_ground_truth_is_not_retrieval_api(self):
        signature = inspect.signature(select_neighbors)
        for name in ("query_distribution", "target", "query_class", "query_gt"):
            self.assertNotIn(name, signature.parameters)
        bank = bank_from_2d([[1, 0]])
        with self.assertRaises(TypeError):
            select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99,
                             query_distribution=torch.tensor([[0., 0., 1., 0.]]))


class EvidenceTests(unittest.TestCase):
    def test_soft_values_hand_computed_and_quality(self):
        bank = bank_from_2d([[1, 0], [0, 1]],
                            labels=[[.1, .2, .7, 0], [.2, .6, .2, 0]],
                            events=[0, 1])
        ids = torch.tensor([[0, 1, -1]])
        scores = torch.tensor([[.5, .5, -torch.inf]])
        evidence = label_evidence(bank, ids, scores)
        expected = torch.tensor([[.15, .4, .45, 0]])
        torch.testing.assert_close(evidence["posterior"], expected)
        self.assertAlmostEqual(evidence["confidence"].item(), .45, places=6)
        self.assertAlmostEqual(evidence["quality"][0, 0].item(), .5, places=6)
        self.assertEqual(evidence["quality"][0, 1].item(), 0)
        self.assertAlmostEqual(evidence["quality"][0, 2].item(), 2 / 3, places=6)
        self.assertAlmostEqual(evidence["quality"][0, 4].item(), .65, places=6)
        self.assertEqual(evidence["quality"][0, 5].item(), 1)

    def test_similarity_temperature_has_expected_weight(self):
        bank = bank_from_2d([[1, 0], [0, 1]], labels=[[0, 1, 0, 0], [0, 0, 1, 0]])
        ids = torch.tensor([[0, 1]])
        scores = torch.tensor([[.2, 0.]])
        evidence = label_evidence(bank, ids, scores, temperature=.1)
        expected = torch.softmax(torch.tensor([2., 0.]), 0)
        torch.testing.assert_close(evidence["posterior"][0, 1:3], expected)
        self.assertAlmostEqual(evidence["quality"][0, 1].item(), .2, places=6)
        with self.assertRaises(ValueError):
            label_evidence(bank, ids, scores, temperature=0)

    def test_event_diversity_counts_distinct_selected_events(self):
        bank = bank_from_2d([[1, 0], [1, 0], [1, 0]], events=[0, 0, 1])
        evidence = label_evidence(bank, torch.tensor([[0, 1, 2]]), torch.ones(1, 3))
        self.assertAlmostEqual(evidence["quality"][0, 5].item(), 2 / 3, places=6)

    def test_soft_diagnostic_overlap_is_not_dominant_class_recall(self):
        bank = bank_from_2d([[1, 0], [0, 1]],
                            labels=[[.1, .2, .7, 0], [.2, .6, .2, 0]])
        target = torch.tensor([[0., .25, .75, 0.], [0., 0., 1., 0.]])
        result = paired_retrieval_diagnostics(
            target, bank, torch.tensor([[0, 1], [-1, -1]]),
            torch.tensor([[1., 1.], [-torch.inf, -torch.inf]]))
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["supported_query_count"], 1)
        self.assertAlmostEqual(result["soft_compatibility_sum"], .4375, places=6)
        self.assertAlmostEqual(result["best_neighbor_soft_compatibility_sum"], .575, places=6)
        self.assertEqual(result["dominant_class_covered_count"], 1)
        self.assertAlmostEqual(result["damage_query_mass_sum"], 1.75, places=6)
        self.assertEqual(result["damage_dominant_covered_count"], 1)
        self.assertFalse(any("recall" in name.lower() for name in result))

    def test_random_control_reproducible_excluded_and_same_missing_mask(self):
        bank = bank_from_2d([[1, 0]] * 12, events=[0] * 4 + [1] * 4 + [2] * 4)
        ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", 0, 7, topk=8)
        first, first_scores = matched_random_ids(bank, ids, scores, 0, 7, seed=1)
        second, _ = matched_random_ids(bank, ids, scores, 0, 7, seed=1)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertTrue(torch.equal(first >= 0, ids >= 0))
        torch.testing.assert_close(first_scores, scores)
        selected = first[first >= 0]
        self.assertEqual(selected.unique().numel(), selected.numel())
        self.assertTrue((bank["events"][selected] != 0).all())
        self.assertTrue((bank["images"][selected] != 7).all())

    def test_random_per_class_preserves_quota(self):
        bank = bank_from_2d([[1, 0]] * 12)
        ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", 9, 99,
                                      topk=8, policy="per_class")
        random, _ = matched_random_ids(bank, ids, scores, 9, 99, seed=3,
                                       policy="per_class")
        dominant = bank["distribution"].argmax(-1)[random]
        self.assertEqual(dominant.tolist(), [[0, 0, 1, 1, 2, 2, 3, 3]])

    def test_shuffled_labels_preserve_members_and_label_multiset(self):
        bank = bank_from_2d([[1, 0]] * 12)
        result = shuffled_label_bank(bank, seed=17)
        for name in ("t128", "f256", "events", "images", "regions"):
            self.assertIs(result[name], bank[name])
        torch.testing.assert_close(result["distribution"].sum(0), bank["distribution"].sum(0))
        self.assertFalse(torch.equal(result["distribution"], bank["distribution"]))
        torch.testing.assert_close(result["distribution"],
                                   shuffled_label_bank(bank, seed=17)["distribution"])


if __name__ == "__main__":
    unittest.main()
