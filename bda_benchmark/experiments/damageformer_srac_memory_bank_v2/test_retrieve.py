"""CPU-only contracts for the frozen bank/retrieval factorial."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from .evaluate import (NativeA1LogitsCapture, check_train_shard_ledger,
                       event_class_counts, native_grid_probq, protocol_record,
                       sum_nested_counts)
from .report import factor_effects, read_formal
from .retrieve import (diversify_ids, posthoc_query_label_compatibility,
                       support_diagnostics, support_provenance_changes)


def bank(images, events, labels=None, regions=None):
    n = len(images)
    if labels is None:
        labels = torch.eye(4)[torch.tensor([1] * n)]
    return dict(tokens=torch.ones(n, 2), images=torch.tensor(images),
                events=torch.tensor(events), regions=torch.tensor(
                    list(range(n)) if regions is None else regions),
                distribution=labels.float())


class RetrievalTests(unittest.TestCase):
    def test_diversity_prefers_images_and_events_then_relaxes_deterministically(self):
        source = bank(images=[0, 0, 1, 2, 3, 4, 5, 6, 7],
                      events=[0, 0, 0, 0, 1, 1, 2, 2, 3])
        ranked = torch.arange(9)[None]
        result, stats = diversify_ids(ranked, source, topk=8)
        self.assertEqual(result.tolist(), [[0, 1, 2, 4, 5, 6, 7, 8]])
        self.assertEqual(stats["relaxed_image_slots"], 0)
        self.assertEqual(stats["relaxed_event_slots"], 1)
        again, repeat = diversify_ids(ranked, source, topk=8)
        self.assertTrue(torch.equal(result, again))
        self.assertEqual(stats, repeat)

    def test_shortlist_shortage_preserves_valid_slots_and_sentinel(self):
        source = bank(images=[0, 0, 1], events=[0, 0, 0])
        ids, stats = diversify_ids(torch.tensor([[0, 1, 2, -1]]), source, topk=8)
        self.assertEqual(ids.tolist(), [[0, 1, 2, -1, -1, -1, -1, -1]])
        self.assertEqual(stats["valid_slots"], 3)
        self.assertEqual(stats["queries_with_shortlist_shortage"], 1)
        self.assertGreater(stats["relaxed_event_slots"], 0)

    def test_invalid_and_duplicate_candidate_ids_fail(self):
        source = bank(images=[0, 1], events=[0, 1])
        with self.assertRaises(ValueError):
            diversify_ids(torch.tensor([[0, 0]]), source)
        with self.assertRaises(ValueError):
            diversify_ids(torch.tensor([[2]]), source)

    def test_provenance_change_uses_source_identity_across_reordered_banks(self):
        old = bank(images=[0, 1, 2], events=[0, 1, 2], regions=[10, 11, 12])
        new = bank(images=[2, 0, 1], events=[2, 0, 1], regions=[12, 10, 11])
        original = torch.tensor([[0, 1, 2], [0, 1, -1]])
        equivalent = torch.tensor([[1, 2, 0], [1, 2, -1]])
        changes = support_provenance_changes(original, old, equivalent, new)
        self.assertEqual(changes["changed_slots"], 0)
        self.assertEqual(changes["queries_with_set_change"], 0)
        reordered = torch.tensor([[2, 1, 0], [1, 2, -1]])
        changes = support_provenance_changes(original, old, reordered, new)
        self.assertEqual(changes["queries_with_rank_change"], 1)
        self.assertEqual(changes["queries_with_set_change"], 0)

    def test_support_diagnostics_and_posthoc_q_mixing(self):
        labels = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0],
                               [0, 0, 1, 0]], dtype=torch.float)
        source = bank(images=[0, 1, 2], events=[0, 1, 2], labels=labels)
        ids = torch.tensor([[1, 2], [0, 1]])
        row = support_diagnostics(ids, source)
        self.assertEqual(row["valid_slots"], 4)
        self.assertEqual(row["unique_image_sum"], 4)
        query = torch.tensor([[0, 1, 0, 0], [0, .5, .5, 0]])
        result = posthoc_query_label_compatibility(ids, source, query,
                                                   torch.ones(2))
        self.assertEqual(result["all"]["queries"], 2)
        self.assertEqual(result["all"]["intact_support_prefers_damaged"], 1)
        self.assertEqual(result["foreground_pure"]["queries"], 1)
        self.assertEqual(result["foreground_mixed"]["queries"], 1)

    def test_no_query_truth_in_retrieval_signature(self):
        import inspect
        from .retrieve import select_diverse
        self.assertFalse(any(name in inspect.signature(select_diverse).parameters
                             for name in ("target", "labels", "query_gt")))


class EvidenceTests(unittest.TestCase):
    def test_native_a1_single_classifier_output_supplies_q_probability(self):
        from torch import nn
        from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(2, 4, 1)

            def forward(self, pre, post):
                native = self.base.clf_2(post)
                return dict(old_pixel=native, q=torch.ones(1, 1, 4),
                            ids=torch.zeros(1, 4, dtype=torch.long), count=1,
                            shape=(2, 2))

        torch.manual_seed(3407)
        front = Front().eval()
        feature_capture = FrozenFrontCapture(front)
        logits_capture = NativeA1LogitsCapture(front)
        try:
            logits_capture.clear()
            result, feature = feature_capture(torch.zeros(1, 2, 2, 2),
                                              torch.ones(1, 2, 2, 2))
            native = logits_capture.take()
            self.assertTrue(torch.equal(feature, torch.ones_like(feature)))
            self.assertTrue(torch.equal(native, result["old_pixel"]))
            prob, replay = native_grid_probq(result, feature, native, (2, 2))
            expected = native.softmax(1).mean(dim=(2, 3), keepdim=True).expand_as(prob)
            self.assertTrue(replay["passed"])
            self.assertTrue(torch.allclose(prob, expected, atol=1e-6))
        finally:
            logits_capture.close()
            feature_capture.close()

    def test_native_a1_capture_rejects_zero_or_multiple_classifier_calls(self):
        from torch import nn

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(2, 4, 1)

        front = Front().eval()
        logits_capture = NativeA1LogitsCapture(front)
        try:
            with self.assertRaisesRegex(RuntimeError, "got 0"):
                logits_capture.take()
            logits_capture.clear()
            front.base.clf_2(torch.zeros(1, 2, 2, 2))
            front.base.clf_2(torch.ones(1, 2, 2, 2))
            with self.assertRaisesRegex(RuntimeError, "got 2"):
                logits_capture.take()
        finally:
            logits_capture.close()

    def test_native_a1_replay_difference_is_reported_and_invalid_data_fails(self):
        logits = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]])
        result = dict(old_pixel=logits + torch.tensor([1.0, 0.0, 0.0, 0.0])
                      .reshape(1, 4, 1, 1), q=torch.ones(1, 1, 1),
                      ids=torch.zeros(1, 1, dtype=torch.long), count=1, shape=(1, 1))
        probability, replay = native_grid_probq(result, torch.zeros(1, 2, 1, 1),
                                                 logits, (1, 1))
        self.assertFalse(replay["passed"])
        self.assertGreater(replay["max_abs"], 2e-5)
        self.assertTrue(torch.allclose(probability, logits.softmax(1)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            native_grid_probq(result, torch.zeros(1, 2, 1, 1),
                              logits * float("nan"), (1, 1))

    def test_count_bucket_and_nested_accumulator(self):
        source = bank(images=[0, 1, 2], events=[0, 1, 1],
                      labels=torch.eye(4)[[1, 2, 2]])
        self.assertEqual(event_class_counts(source, 2),
                         [[0, 1, 0, 0], [0, 0, 2, 0]])
        accumulator = {}
        sum_nested_counts(accumulator, {"a": {"v": [1, 2], "f": .25}})
        sum_nested_counts(accumulator, {"a": {"v": [3, 4], "f": .5}})
        self.assertEqual(accumulator, {"a": {"v": [4, 6], "f": .75}})

    def test_train_shard_ledger_compares_projection_and_identity(self):
        names = [f"eventA_{i}" for i in range(2918)]
        originals = [dict(path=f"shards/{i:05d}.pt", sha256=f"hash-{i}")
                     for i in range(2918)]
        records = [dict(**row, name=names[i], event=0)
                   for i, row in enumerate(originals)]
        check_train_shard_ledger(records, originals, names, ["eventA"],
                                 lambda name: name.split("_")[0])
        records[17]["name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "17"):
            check_train_shard_ledger(records, originals, names, ["eventA"],
                                     lambda name: name.split("_")[0])
        records[17]["name"] = names[17]
        records[17]["sha256"] = "wrong-hash"
        with self.assertRaisesRegex(ValueError, "17"):
            check_train_shard_ledger(records, originals, names, ["eventA"],
                                     lambda name: name.split("_")[0])

    def test_protocol_rejects_failed_or_strict_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(dict(format="SRAC_HARDQUERY_PROTOCOL_V1",
                                            complete=True,
                                            full_validation_evidence=True,
                                            reproduction={"passed": True}, image_count=450,
                                            scope="all", smoke=False, profile="strict",
                                            pool_mode="native", selected_images=[str(i)
                                            for i in range(450)])), encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol_record(path)

    def test_report_rejects_nonformal_and_factor_effect_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "summary.json").write_text(json.dumps({"complete": False,
                                                               "audit_file": "audit.json"}),
                                                 encoding="utf-8")
            (output / "audit.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_formal(output)
        audit = dict(metrics={"j3": {"valid_miou": .60},
                              "j3_old_diverse": {"valid_miou": .61},
                              "j3_quality_top8": {"valid_miou": .62},
                              "j3_quality_diverse": {"valid_miou": .64}})
        effects = factor_effects(audit)
        self.assertAlmostEqual(effects["quality_bank_effect_at_normal_top8_pp"], 2)
        self.assertAlmostEqual(effects["diverse_top8_effect_on_old_bank_pp"], 1)
        self.assertAlmostEqual(effects["bank_selection_x_retrieval_interaction_pp"], 1)


if __name__ == "__main__":
    unittest.main()
