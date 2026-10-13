"""Independent dense-reference and hand-calculated FP32 retrieval contracts."""

import inspect
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from . import core
from .core import (label_evidence, matched_random_ids, pair_neighbor_overlap,
                   paired_retrieval_diagnostics, select_neighbors,
                   shuffled_label_bank, validate_paired_bank)


def bank_from_2d(vectors, labels=None, events=None, images=None):
    values = torch.tensor(vectors, dtype=torch.float32).reshape(-1, 2)
    rows = values.shape[0]
    t128, f256 = torch.zeros(rows, 128), torch.zeros(rows, 256)
    t128[:, :2], f256[:, :2] = values, values
    if labels is None:
        labels = torch.nn.functional.one_hot(torch.arange(rows) % 4, 4).float()
    else:
        labels = torch.tensor(labels, dtype=torch.float32).reshape(rows, 4)
    events = torch.tensor(events if events is not None else [0] * rows, dtype=torch.long)
    images = torch.tensor(images if images is not None else list(range(rows)), dtype=torch.long)
    return {"t128": t128, "f256": f256, "distribution": labels,
            "events": events, "images": images, "regions": torch.arange(rows),
            "original_rows": torch.arange(rows), "member_ids": torch.arange(rows),
            "event_names": [f"event_{i}" for i in range(
                int(events.max()) + 1 if rows else 0)], "split": "Train"}


def query2d(values, key="t128"):
    value = torch.tensor(values, dtype=torch.float32).reshape(-1, 2)
    query = torch.zeros(value.shape[0], 128 if key == "t128" else 256)
    query[:, :2] = value
    return query


def independent_dense_reference(query, bank, key, events, images, topk):
    """Double-precision scalar cosine and Python sorting; no core helpers."""
    keys = bank[key].double()
    queries = query.double()
    ids, scores = [], []
    for qrow, q in enumerate(queries):
        candidates = []
        for member, feature in enumerate(keys):
            norms = [float(bank[name][member].double().norm()) for name in ("t128", "f256")]
            if min(norms) <= 1e-8 or float(q.norm()) <= 1e-8:
                continue
            if int(bank["events"][member]) == int(events[qrow]):
                continue
            if int(bank["images"][member]) == int(images[qrow]):
                continue
            cosine = float(torch.dot(q, feature) / (q.norm() * feature.norm()))
            candidates.append((cosine, member))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = candidates[:topk]
        ids.append([member for _, member in chosen] + [-1] * (topk - len(chosen)))
        scores.append([score for score, _ in chosen] + [-math.inf] * (topk - len(chosen)))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


class BankValidationTests(unittest.TestCase):
    def test_valid_fp32_bank_and_empty_bank(self):
        validate_paired_bank(bank_from_2d([[1, 0], [0, 1]]))
        validate_paired_bank(bank_from_2d([]))

    def test_each_float_artifact_rejects_half_and_double(self):
        bank = bank_from_2d([[1, 0]])
        for name in ("t128", "f256", "distribution"):
            for dtype in (torch.float16, torch.float64):
                with self.subTest(name=name, dtype=dtype):
                    bad = dict(bank, **{name: bank[name].to(dtype)})
                    with self.assertRaisesRegex(ValueError, "float32"):
                        validate_paired_bank(bad)

    def test_provenance_requires_long_and_nonnegative(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        for name in ("events", "images", "regions", "original_rows", "member_ids"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "torch.long"):
                    validate_paired_bank(dict(bank, **{name: bank[name].int()}))
                with self.assertRaisesRegex(ValueError, "negative"):
                    validate_paired_bank(dict(bank, **{name: -torch.ones(2, dtype=torch.long)}))

    def test_stable_member_identifiers_cannot_be_duplicate(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        for name in ("original_rows", "member_ids"):
            with self.assertRaisesRegex(ValueError, "unique"):
                validate_paired_bank(dict(bank, **{name: torch.zeros(2, dtype=torch.long)}))
        legacy_minimal = {name: value for name, value in bank.items()
                          if name not in ("original_rows", "member_ids")}
        validate_paired_bank(legacy_minimal)

    def test_event_names_indices_and_split_flags(self):
        bank = bank_from_2d([[1, 0]], events=[1])
        for names in (["same", "same"], ["one"], ["", "two"], [0, "two"]):
            with self.assertRaises(ValueError):
                validate_paired_bank(dict(bank, event_names=names))
        for split in ("Val", "Test", "all"):
            with self.assertRaisesRegex(ValueError, "Train"):
                validate_paired_bank(dict(bank, split=split))

    def test_nonfinite_and_malformed_soft_labels_fail(self):
        bank = bank_from_2d([[1, 0]])
        for key in ("t128", "f256"):
            for nonfinite in (float("nan"), float("inf")):
                corrupt = bank[key].clone()
                corrupt[0, 0] = nonfinite
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    validate_paired_bank(dict(bank, **{key: corrupt}))
        for labels in ([[float("nan"), 0, 0, 1]], [[-.1, .1, 0, 1]],
                       [[0, 0, 0, .5]]):
            with self.assertRaises(ValueError):
                validate_paired_bank(dict(bank, distribution=torch.tensor(labels)))
        with self.assertRaisesRegex(ValueError, "f256"):
            validate_paired_bank(dict(bank, f256=torch.zeros(1, 255)))


class RetrievalTests(unittest.TestCase):
    def test_dense_reference_top8_top64_both_spaces_all_chunk_sizes(self):
        generator = torch.Generator().manual_seed(73)
        members, queries = 93, 11
        bank = bank_from_2d([[1, 0]] * members,
                            events=(torch.arange(members) % 7).tolist(),
                            images=(torch.arange(members) % 29).tolist())
        bank["t128"] = torch.randn(members, 128, generator=generator)
        bank["f256"] = torch.randn(members, 256, generator=generator)
        bank["t128"][5] = 0
        bank["f256"][6] = 0
        events = torch.arange(queries) % 7
        images = torch.arange(queries) % 29
        for key, width in (("t128", 128), ("f256", 256)):
            query = torch.randn(queries, width, generator=generator)
            query[2] = 0
            for topk in (8, 64):
                expected_ids, expected_scores = independent_dense_reference(
                    query, bank, key, events, images, topk)
                for chunk in (1, 3, 256):
                    with self.subTest(key=key, topk=topk, chunk=chunk):
                        ids, scores = select_neighbors(query, bank, key, events, images,
                                                       topk=topk, chunk_size=chunk)
                        self.assertTrue(torch.equal(ids, expected_ids))
                        torch.testing.assert_close(scores, expected_scores,
                                                   atol=2e-6, rtol=2e-6)

    def test_exact_cosines_stable_ties_and_padding(self):
        bank = bank_from_2d([[1, 0], [3, 0], [3, 4], [-1, 0], [0, 2]])
        bank["original_rows"] = torch.tensor([15, 8, 4, 3, 9])
        ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", -1, -1, topk=8)
        self.assertEqual(ids.tolist(), [[0, 1, 2, 4, 3, -1, -1, -1]])
        torch.testing.assert_close(scores, torch.tensor([[1., 1., .6, 0., -1.,
                                                        -torch.inf, -torch.inf, -torch.inf]]))

    def test_same_event_and_same_image_exclusions_are_independent(self):
        bank = bank_from_2d([[1, 0], [3, 4], [0, 1], [-1, 0]],
                            events=[0, 1, 1, 2], images=[10, 11, 12, 13])
        ids, _ = select_neighbors(query2d([[1, 0], [1, 0]]), bank, "t128",
                                  torch.tensor([0, 1]), torch.tensor([11, 13]),
                                  topk=3, chunk_size=1)
        self.assertEqual(ids.tolist(), [[2, 3, -1], [0, -1, -1]])

    def test_joint_member_validity_and_zero_query_fallback(self):
        bank = bank_from_2d([[1, 0], [0, 1], [-1, 0], [3, 4]])
        bank["f256"][0] = 0
        bank["t128"][1] = 0
        for key in ("t128", "f256"):
            ids, scores = select_neighbors(query2d([[1, 0], [0, 0]], key),
                                           bank, key, -1, -1, topk=3)
            self.assertEqual(ids.tolist(), [[3, 2, -1], [-1, -1, -1]])
            self.assertTrue(torch.isneginf(scores[1]).all())

    def test_empty_excluded_and_zero_query_support_never_nan(self):
        for bank, event in ((bank_from_2d([]), -1),
                            (bank_from_2d([[1, 0]]), 0)):
            ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", event, -1)
            evidence = label_evidence(bank, ids, scores)
            self.assertEqual(ids.tolist(), [[-1] * 8])
            self.assertFalse(evidence["supported"].item())
            self.assertEqual(evidence["posterior"].tolist(), [[0.] * 4])
            self.assertEqual(evidence["quality"].tolist(), [[0.] * 6])
            self.assertTrue(torch.isfinite(evidence["weights"]).all())

    def test_empty_query_batch_preserves_shapes(self):
        ids, scores = select_neighbors(torch.zeros(0, 128), bank_from_2d([]),
                                       "t128", -1, -1, topk=64)
        self.assertEqual(ids.shape, (0, 64))
        self.assertEqual(label_evidence(bank_from_2d([]), ids, scores)["posterior"].shape,
                         (0, 4))

    def test_paired_key_spaces_can_change_ranking_without_changing_labels(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        bank["f256"][:, :2] = torch.tensor([[0., 1.], [1., 0.]])
        labels = bank["distribution"].clone()
        t_ids, _ = select_neighbors(query2d([[1, 0]]), bank, "t128", -1, -1, topk=1)
        f_ids, _ = select_neighbors(query2d([[1, 0]], "f256"), bank, "f256", -1, -1, topk=1)
        self.assertEqual(t_ids.tolist(), [[0]])
        self.assertEqual(f_ids.tolist(), [[1]])
        self.assertTrue(torch.equal(bank["distribution"], labels))

    def test_query_gt_absent_from_api_and_support_label_changes_do_not_rank(self):
        signature = inspect.signature(select_neighbors)
        for name in ("query_distribution", "target", "query_class", "query_gt"):
            self.assertNotIn(name, signature.parameters)
        bank = bank_from_2d([[1, 0], [3, 4], [0, 1]])
        query = query2d([[1, 0]])
        before = select_neighbors(query, bank, "t128", -1, -1, topk=3)
        bank["distribution"] = torch.tensor([[0., 0., 0., 1.]] * 3)
        after = select_neighbors(query, bank, "t128", -1, -1, topk=3)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))
        with self.assertRaises(TypeError):
            select_neighbors(query, bank, "t128", -1, -1, query_gt=torch.tensor([2]))

    def test_nonfinite_query_invalid_controls_and_global_only(self):
        bank = bank_from_2d([[1, 0]])
        for invalid in (float("nan"), float("inf")):
            query = query2d([[invalid, 0]])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                select_neighbors(query, bank, "t128", -1, -1)
        for kwargs in ({"topk": 0}, {"chunk_size": 0}, {"policy": "per_class"}):
            with self.assertRaises(ValueError):
                select_neighbors(query2d([[1, 0]]), bank, "t128", -1, -1, **kwargs)
        with self.assertRaisesRegex(ValueError, "float32"):
            select_neighbors(query2d([[1, 0]]).half(), bank, "t128", -1, -1)
        with self.assertRaisesRegex(ValueError, "sentinel"):
            select_neighbors(query2d([[1, 0]]), bank, "t128", -2, -1)

    def test_large_finite_features_normalize_without_overflow(self):
        bank = bank_from_2d([[3e38, 3e38], [-3e38, -3e38]])
        ids, scores = select_neighbors(query2d([[3e38, 3e38]]), bank,
                                       "t128", -1, -1, topk=2)
        self.assertEqual(ids.tolist(), [[0, 1]])
        torch.testing.assert_close(scores, torch.tensor([[1., -1.]]))


class LabelEvidenceTests(unittest.TestCase):
    def test_equal_weight_soft_labels_and_hand_computed_quality(self):
        bank = bank_from_2d([[1, 0], [0, 1]],
                            labels=[[.1, .2, .7, 0], [.2, .6, .2, 0]], events=[0, 1])
        evidence = label_evidence(bank, torch.tensor([[0, 1, -1]]),
                                  torch.tensor([[.5, .5, -torch.inf]]))
        torch.testing.assert_close(evidence["posterior"], torch.tensor([[.15, .4, .45, 0]]))
        self.assertAlmostEqual(evidence["confidence"].item(), .45, places=6)
        self.assertAlmostEqual(evidence["quality"][0, 0].item(), .5, places=6)
        self.assertEqual(evidence["quality"][0, 1].item(), 0)
        self.assertAlmostEqual(evidence["quality"][0, 2].item(), 2 / 3, places=6)
        self.assertAlmostEqual(evidence["quality"][0, 4].item(), .65, places=6)
        self.assertEqual(evidence["quality"][0, 5].item(), 1)

    def test_nonuniform_vote_uses_cosine_temperature_arithmetic(self):
        bank = bank_from_2d([[1, 0], [0, 1]], labels=[[0, 1, 0, 0], [0, 0, 1, 0]])
        evidence = label_evidence(bank, torch.tensor([[0, 1]]), torch.tensor([[.2, 0.]]), .1)
        weight = math.exp(2) / (math.exp(2) + 1)
        self.assertAlmostEqual(evidence["posterior"][0, 1].item(), weight, places=6)
        self.assertAlmostEqual(evidence["posterior"][0, 2].item(), 1 - weight, places=6)
        self.assertAlmostEqual(evidence["quality"][0, 1].item(), .2, places=6)

    def test_tiny_positive_temperature_is_finite_and_selects_maximum(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        evidence = label_evidence(bank, torch.tensor([[0, 1]]),
                                  torch.tensor([[.9, -.9]]), temperature=1e-300)
        self.assertTrue(torch.isfinite(evidence["posterior"]).all())
        torch.testing.assert_close(evidence["posterior"], bank["distribution"][:1])
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                label_evidence(bank, torch.tensor([[0]]), torch.tensor([[.9]]), temperature)

    def test_zero_slots_and_empty_batch_are_safe(self):
        bank = bank_from_2d([[1, 0]])
        evidence = label_evidence(bank, torch.empty(2, 0, dtype=torch.long), torch.empty(2, 0))
        self.assertEqual(evidence["posterior"].tolist(), [[0.] * 4] * 2)
        evidence = label_evidence(bank, torch.empty(0, 8, dtype=torch.long), torch.empty(0, 8))
        self.assertEqual(evidence["posterior"].shape, (0, 4))

    def test_corrupt_selected_ids_and_nonfinite_selected_scores_fail(self):
        bank = bank_from_2d([[1, 0]])
        for ids in (torch.tensor([[1]]), torch.tensor([[-2]]), torch.tensor([[0]], dtype=torch.int32)):
            with self.assertRaises(ValueError):
                label_evidence(bank, ids, torch.tensor([[.5]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            label_evidence(bank, torch.tensor([[0]]), torch.tensor([[float("nan")]]))

    def test_event_diversity_counts_distinct_valid_events(self):
        bank = bank_from_2d([[1, 0]] * 3, events=[0, 0, 1])
        evidence = label_evidence(bank, torch.tensor([[0, 1, 2, -1]]),
                                  torch.tensor([[1., 1., 1., -torch.inf]]))
        self.assertAlmostEqual(evidence["quality"][0, 5].item(), 2 / 3, places=6)

    def test_posthoc_soft_overlap_agreement_and_conflict_are_distinct(self):
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
        self.assertEqual(result["dominant_label_conflict_count"], 0)
        conflict = paired_retrieval_diagnostics(
            torch.tensor([[0., 1., 0., 0.]]), bank,
            torch.tensor([[0]]), torch.tensor([[1.]]))
        self.assertEqual(conflict["dominant_label_conflict_count"], 1)
        self.assertFalse(any("recall" in name.lower() for name in result))

    def test_damage_destroyed_mass_combines_classes_2_and_3(self):
        bank = bank_from_2d([[1, 0]], labels=[[0, .2, .3, .5]])
        report = paired_retrieval_diagnostics(torch.tensor([[0., 0., 0., 1.]]), bank,
                                              torch.tensor([[0]]), torch.tensor([[1.]]))
        self.assertEqual(report["damage_query_mass_sum"], 0)
        self.assertEqual(report["damaged_destroyed_query_mass_sum"], 1)
        self.assertAlmostEqual(report["damaged_destroyed_evidence_mass_sum"], .8, places=6)
        self.assertAlmostEqual(report["damaged_destroyed_shared_mass_sum"], .8, places=6)

    def test_diagnostics_are_additive_across_query_chunks(self):
        bank = bank_from_2d([[1, 0], [0, 1]])
        target = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
        ids, scores = torch.tensor([[0, 1], [-1, -1]]), torch.tensor([[.8, .6], [-torch.inf, -torch.inf]])
        whole = paired_retrieval_diagnostics(target, bank, ids, scores)
        pieces = [paired_retrieval_diagnostics(target[i:i+1], bank, ids[i:i+1], scores[i:i+1])
                  for i in range(2)]
        for key, value in whole.items():
            if isinstance(value, list):
                expected = torch.tensor(pieces[0][key]) + torch.tensor(pieces[1][key])
                torch.testing.assert_close(torch.tensor(value), expected)
            else:
                self.assertAlmostEqual(value, pieces[0][key] + pieces[1][key], places=6)


class NeighborComparisonTests(unittest.TestCase):
    def test_overlap_top1_and_empty_pairs_are_hand_computed(self):
        a = torch.tensor([[0, 1, 2], [1, -1, -1], [-1, -1, -1]])
        b = torch.tensor([[0, 2, 3], [2, -1, -1], [-1, -1, -1]])
        report = pair_neighbor_overlap(a, b)
        self.assertEqual(report["query_count"], 3)
        self.assertEqual(report["both_supported_query_count"], 2)
        self.assertEqual(report["intersection_slot_count"], 2)
        self.assertEqual(report["union_slot_count"], 6)
        self.assertAlmostEqual(report["neighbor_jaccard_sum"], .5)
        self.assertAlmostEqual(report["neighbor_overlap_fraction_sum"], 2 / 3, places=6)
        self.assertEqual(report["top1_agreement_count"], 1)

    def test_overlap_different_slot_counts_and_empty_arrays(self):
        report = pair_neighbor_overlap(torch.tensor([[0, 1]]), torch.tensor([[1]]))
        self.assertEqual(report["intersection_slot_count"], 1)
        self.assertEqual(report["union_slot_count"], 2)
        report = pair_neighbor_overlap(torch.empty(2, 0, dtype=torch.long),
                                       torch.empty(2, 0, dtype=torch.long))
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["neighbor_jaccard_sum"], 0)

    def test_overlap_rejects_duplicate_and_invalid_members(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pair_neighbor_overlap(torch.tensor([[1, 1]]), torch.tensor([[1, 2]]))
        with self.assertRaises(ValueError):
            pair_neighbor_overlap(torch.tensor([[-2]]), torch.tensor([[0]]))

    def test_random_compatibility_helper_retains_scores_not_cosines(self):
        bank = bank_from_2d([[1, 0]] * 12, events=[0] * 4 + [1] * 4 + [2] * 4)
        ids, scores = select_neighbors(query2d([[1, 0]]), bank, "t128", 0, 7, topk=8)
        first, first_scores = matched_random_ids(bank, ids, scores, 0, 7, seed=1)
        second, _ = matched_random_ids(bank, ids, scores, 0, 7, seed=1)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first >= 0, ids >= 0))
        self.assertTrue(torch.equal(first_scores, scores))
        selected = first[first >= 0]
        self.assertEqual(selected.unique().numel(), selected.numel())
        self.assertTrue((bank["events"][selected] != 0).all())
        self.assertTrue((bank["images"][selected] != 7).all())

    def test_shuffled_compatibility_helper_preserves_keys_and_label_multiset(self):
        bank = bank_from_2d([[1, 0]] * 12)
        result = shuffled_label_bank(bank, seed=17)
        for name in ("t128", "f256", "events", "images", "regions", "member_ids"):
            self.assertIs(result[name], bank[name])
        torch.testing.assert_close(result["distribution"].sum(0), bank["distribution"].sum(0))
        self.assertFalse(torch.equal(result["distribution"], bank["distribution"]))


class RetrievalPrecisionTests(unittest.TestCase):
    def _cuda_operation(self, original_precision, original_tf32, raises=False):
        flags = SimpleNamespace(allow_tf32=original_tf32)
        precision = [original_precision]
        observed = []

        def set_precision(value):
            precision[0] = value
            flags.allow_tf32 = value != "highest"

        class FakeCudaQuery:
            device = torch.device("cuda")

            def __matmul__(inner_self, other):
                observed.append((precision[0], flags.allow_tf32))
                if raises:
                    raise RuntimeError("injected CUDA score failure")
                return torch.tensor([[1., 0.]]) @ other

        with mock.patch.object(core.torch.backends.cuda, "matmul", flags), \
             mock.patch.object(core.torch, "get_float32_matmul_precision",
                               side_effect=lambda: precision[0]), \
             mock.patch.object(core.torch, "set_float32_matmul_precision",
                               side_effect=set_precision) as setter:
            if raises:
                with self.assertRaisesRegex(RuntimeError, "injected CUDA"):
                    core._cosine_dot(FakeCudaQuery(), torch.eye(2))
            else:
                scores = core._cosine_dot(FakeCudaQuery(), torch.eye(2))
                torch.testing.assert_close(scores, torch.tensor([[1., 0.]]))
            self.assertEqual(observed, [("highest", False)])
            self.assertEqual(precision[0], original_precision)
            self.assertEqual(flags.allow_tf32, original_tf32)
            self.assertEqual(setter.call_args_list,
                             [mock.call("highest"), mock.call(original_precision)])

    def test_cuda_score_uses_full_fp32_then_restores_runtime_policy(self):
        for precision, tf32 in (("highest", False), ("high", True), ("medium", True)):
            with self.subTest(precision=precision, tf32=tf32):
                self._cuda_operation(precision, tf32)

    def test_cuda_score_failure_restores_runtime_policy_and_propagates(self):
        self._cuda_operation("high", True, raises=True)

    def test_cpu_score_never_reads_or_writes_cuda_precision_flags(self):
        class ForbiddenFlags:
            @property
            def allow_tf32(self):
                raise AssertionError("CPU retrieval touched CUDA flags")

            @allow_tf32.setter
            def allow_tf32(self, value):
                raise AssertionError("CPU retrieval changed CUDA flags")

        with mock.patch.object(core.torch.backends.cuda, "matmul", ForbiddenFlags()), \
             mock.patch.object(core.torch, "get_float32_matmul_precision",
                               side_effect=AssertionError("CPU read global precision")), \
             mock.patch.object(core.torch, "set_float32_matmul_precision",
                               side_effect=AssertionError("CPU changed global precision")):
            scores = core._cosine_dot(torch.tensor([[1., 0.]]), torch.eye(2))
            torch.testing.assert_close(scores, torch.tensor([[1., 0.]]))

    def _actual_torch_runtime_policy(self, raises):
        """Use the real coupled Torch setters; only the CUDA tensor is fake."""
        caller_precision = torch.get_float32_matmul_precision()
        caller_tf32 = torch.backends.cuda.matmul.allow_tf32
        test = self

        class FakeCudaQuery:
            device = torch.device("cuda")

            def __matmul__(inner_self, other):
                test.assertEqual(torch.get_float32_matmul_precision(), "highest")
                test.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                if raises:
                    raise RuntimeError("actual-policy injected score failure")
                return torch.tensor([[1., 0.]]) @ other

        try:
            for precision in ("highest", "high", "medium"):
                with self.subTest(precision=precision, raises=raises):
                    torch.set_float32_matmul_precision(precision)
                    before = (torch.get_float32_matmul_precision(),
                              torch.backends.cuda.matmul.allow_tf32)
                    self.assertEqual(before, (precision, precision != "highest"))
                    if raises:
                        with self.assertRaisesRegex(RuntimeError, "actual-policy"):
                            core._cosine_dot(FakeCudaQuery(), torch.eye(2))
                    else:
                        scores = core._cosine_dot(FakeCudaQuery(), torch.eye(2))
                        torch.testing.assert_close(scores, torch.tensor([[1., 0.]]))
                    after = (torch.get_float32_matmul_precision(),
                             torch.backends.cuda.matmul.allow_tf32)
                    self.assertEqual(after, before)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = caller_tf32
            torch.set_float32_matmul_precision(caller_precision)

    def test_actual_torch_precision_backend_coupling_restores_all_three_policies(self):
        self._actual_torch_runtime_policy(raises=False)

    def test_actual_torch_precision_backend_coupling_restores_on_score_failure(self):
        self._actual_torch_runtime_policy(raises=True)


if __name__ == "__main__":
    unittest.main()
