"""Real CPU forward/retrieval integration, GT ordering and persisted evidence."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
from . import core, paired, shared
from .diagnose import (ALL_ARMS, RETRIEVAL_ARMS, _record_failure, _verify_saved_artifacts,
                       _vote_probability, _weighted_evidence_diagnostic, formal_gate, run)
from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _integration(self, *, names_count=450, temperature=.1, limit=1, count=2):
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import memory_audit
        torch.manual_seed(44)
        feature = torch.randn(1, 256, 1, 3)
        tokens = torch.randn(1, 3, 128)
        tokens[:, 2] = 0
        q = torch.full((1, 9, 3), 1 / 9)
        ids = torch.tensor([[0, 1, 2]]).expand(9, -1).clone()
        mass = torch.zeros(1, 3)
        for slot in range(9):
            mass.scatter_add_(1, ids[slot:slot+1], q[:, slot])
        calls = []

        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(256, 4, 1)
            def forward(self, pre, post):
                calls.append("front")
                logits = self.base.clf_2(feature)
                return dict(tokens=tokens, old_pixel=logits, prob=logits.softmax(1),
                    q=q, ids=ids, mass=mass, count=3, shape=(1, 3),
                    valid_regions=torch.tensor([[True, True, False]]))

        bank = dict(t128=torch.randn(6, 128), f256=torch.randn(6, 256),
            distribution=torch.eye(4)[torch.tensor([0, 1, 2, 3, 1, 2])].clone(),
            events=torch.ones(6, dtype=torch.long), images=torch.arange(6),
            regions=torch.zeros(6, dtype=torch.long), event_names=["val", "support"],
            original_rows=torch.arange(6), member_ids=torch.arange(6), split="train",
            label_mass=torch.ones(6), eligible=torch.ones(6, dtype=torch.bool))
        names = [f"val_image_{index}" for index in range(names_count)]
        dataset = self.root / "dataset"
        for folder in ("pre-event", "post-event", "target"):
            (dataset / folder).mkdir(parents=True, exist_ok=True)
        for name in names[:limit or len(names)]:
            for folder, suffix in (("pre-event", "_pre_disaster.tif"), ("post-event", "_post_disaster.tif"),
                                    ("target", "_building_damage.tif")):
                (dataset / folder / (name + suffix)).write_bytes(name.encode())
        bundle = dict(args=SimpleNamespace(dataset=str(dataset), workers=0, seed=3407, pool_mode="native"),
            val_names=names, train_names=[f"train_{i}" for i in range(2918)], device=torch.device("cpu"),
            front=Front(), protocol_sha256="source_anchor")
        key_root = self.root / "keys"
        key_root.mkdir(exist_ok=True)
        for name in ("paired_bank.pt", "manifest.json", "summary.json", "protocol.json", "images.jsonl"):
            (key_root / name).write_bytes(b"external fixture")
        targets = [torch.tensor([[[1, 2, 255]]]), torch.tensor([[[3, 1, 0]]])]
        batches = [(torch.empty(1, 3, 1, 3), torch.empty(1, 3, 1, 3), None, targets[i % 2], [names[i]])
                   for i in range(min(count, len(names), limit or len(names)))]
        real_select, real_labels = core.select_neighbors, paired.extract_labels
        def select(*args, **kwargs):
            last_front = max(i for i, value in enumerate(calls) if value == "front")
            self.assertNotIn("gt", calls[last_front:])
            self.assertEqual(kwargs["topk"], 64)
            self.assertEqual(kwargs["policy"], "global")
            calls.append("neighbors")
            return real_select(*args, **kwargs)
        def labels(*args, **kwargs):
            self.assertEqual(calls[-3:], ["front", "neighbors", "neighbors"])
            calls.append("gt")
            return real_labels(*args, **kwargs)
        output = self.root / "diagnostic"
        with mock.patch.object(shared, "load_bundle", return_value=bundle), \
             mock.patch.object(shared, "load_paired_bank", return_value=(bank, {"smoke": True, "rebuild_protocol_sha256": "new_protocol"})), \
             mock.patch.object(shared, "verify_all", return_value={"fixtures": True}), \
             mock.patch.object(memory_audit, "full_image_loader", return_value=batches), \
             mock.patch.object(memory_audit, "parse_batch", side_effect=lambda batch, _device: batch), \
             mock.patch.object(memory_audit, "event", return_value="val"), \
             mock.patch.object(core, "select_neighbors", side_effect=select), \
             mock.patch.object(paired, "extract_labels", side_effect=labels):
            result = run(SimpleNamespace(protocol="source_anchor", keys=str(key_root),
                output=str(output), limit=limit, temperature=temperature))
        return result, calls, bank, output, targets

    def test_real_cpu_smoke_one_front_and_gt_after_both_key_rankings(self):
        result, calls, bank, output, _ = self._integration()
        self.assertEqual(calls, ["front", "neighbors", "neighbors", "gt"])
        self.assertTrue(result["execution_complete"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["historical_reproduction_claim"])
        self.assertEqual(result["query_eligibility"]["inactive_both_keys"], 1)
        self.assertEqual(result["query_eligibility"]["unlabeled_posthoc_excluded"], 1)
        self.assertTrue(result["pool_guard"]["restored"])
        with np.load(output / "predictions" / "0000.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["query_t128"].dtype, np.float32)
            self.assertEqual(saved["query_f256"].dtype, np.float32)
            for arm in RETRIEVAL_ARMS:
                self.assertTrue((saved["support_" + arm][2] == -1).all())
                self.assertTrue((saved["evidence_" + arm][2] == 0).all())
            for key in ("t128", "f256"):
                self.assertTrue(np.array_equal(saved[f"support_{key}_top8"], saved[f"support_{key}_top64"][:, :8]))
        self.assertEqual(set(result["metrics"]), set(ALL_ARMS))
        self.assertEqual(result["query_forward_count"], 1)
        self.assertEqual(result["historical_reader_forward_count"], 0)
        self.assertEqual(result["bank_label_inventory"]["eligible"]["support_count"], 6)
        self.assertEqual(result["bank_label_inventory"]["eligible"]["dominant_counts"], [1, 2, 2, 1])

    def test_real_cpu_two_images_reopened_predictions_match_whole_split_metrics(self):
        result, calls, bank, output, _ = self._integration(limit=2)
        self.assertEqual(calls.count("front"), 2)
        for arm, cm in result["confusion"].items():
            self.assertEqual(cm, result["metrics"][arm]["confusion"])
            self.assertEqual(result["metrics"][arm]["count"], 5)
            arr = np.asarray(cm, dtype=float)
            union = arr.sum(0) + arr.sum(1) - np.diag(arr)
            expected = (np.diag(arr) / np.maximum(union, 1))[1:].mean()
            self.assertAlmostEqual(result["metrics"][arm]["valid_miou"], expected)
        _verify_saved_artifacts(output, ["val_image_0", "val_image_1"], bank, result["confusion"])

    def test_missing450_rejects_before_model_forward(self):
        with self.assertRaisesRegex(ValueError, "Train2918/Val450"):
            self._integration(names_count=449)
        summary = json.loads((self.root / "diagnostic" / "summary.json").read_text())
        self.assertEqual(summary["image_count"], 0)
        self.assertEqual(summary["phase"], "load")

    def test_temperature_is_bound_to_new_protocol(self):
        with self.assertRaisesRegex(ValueError, "fixes.*0.1"):
            self._integration(temperature=.2)
        self.assertFalse((self.root / "diagnostic").exists())

    def test_formal_gate_requires_every_boolean_and_full450_non_smoke_bank(self):
        checks = {key: True for key in ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")}
        self.assertTrue(formal_gate(450, 450, checks))
        for key in checks:
            for invalid in (False, None, 1):
                modified = {**checks, key: invalid}
                self.assertFalse(formal_gate(450, 450, modified))
        self.assertFalse(formal_gate(2, 450, checks))
        self.assertFalse(formal_gate(450, 450, checks, smoke=True))
        self.assertFalse(formal_gate(450, 450, checks, bank_smoke=True))

    def test_unsupported_vote_exactly_preserves_grid_probq(self):
        baseline = torch.tensor([[[[.1, .2]], [[.2, .3]], [[.3, .4]], [[.4, .1]]]])
        result = dict(q=torch.ones(1, 1, 2), ids=torch.tensor([[0, 1]]), shape=(1, 2))
        evidence = dict(posterior=torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]]),
                        supported=torch.tensor([True, False]))
        probability, coverage = _vote_probability(evidence, result, (1, 2), baseline)
        self.assertTrue(torch.equal(probability[0, :, 0, 0], evidence["posterior"][0]))
        self.assertTrue(torch.equal(probability[0, :, 0, 1], baseline[0, :, 0, 1]))
        self.assertTrue(torch.equal(coverage, torch.tensor([[[1., 0.]]])))

    def test_posthoc_partitions_include_destroyed_class_as_damage(self):
        evidence = dict(posterior=torch.tensor([[0., 0., 0., 1.], [0., .5, .5, 0.], [0., 0., 0., 0.]]),
                        supported=torch.tensor([True, True, False]))
        query = torch.tensor([[0., 1., 0., 0.], [0., .5, .5, 0.], [0., 0., 1., 0.]])
        row = _weighted_evidence_diagnostic(evidence, query, torch.tensor([2., 4., 1.]))
        self.assertEqual(row["all"]["intact_vote_damaged_or_destroyed"], 1)
        self.assertEqual(row["foreground_mixed"]["queries"], 1)
        self.assertAlmostEqual(row["all"]["supported_valid_mass"], 6.)

    def _mutate(self, field, mutate, error_match):
        result, _calls, bank, output, _ = self._integration()
        artifact = output / "predictions" / "0000.npz"
        with np.load(artifact, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays[field] = mutate(arrays[field])
        np.savez_compressed(artifact, **arrays)
        row = json.loads((output / "images.jsonl").read_text())
        row["prediction_sha256"] = digest(artifact)
        (output / "images.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, error_match):
            _verify_saved_artifacts(output, ["val_image_0"], bank, result["confusion"])

    def test_valid_new_checksum_cannot_hide_changed_cosine(self):
        self._mutate("scores_t128_top64", lambda a: a + np.where(np.isfinite(a), .2, 0), "prefix|cosine")

    def test_valid_new_checksum_cannot_hide_changed_train_values(self):
        self._mutate("softlabels_f256_top8", lambda a: np.zeros_like(a), "Train bank")

    def test_valid_new_checksum_cannot_hide_changed_pixel_prediction(self):
        self._mutate("prediction_a1_pixel", lambda a: (a + 1) % 4, "confusion")

    def test_artifact_hash_journal_rejects_corruption(self):
        _result, _calls, bank, output, _ = self._integration()
        path = output / "predictions" / "0000.npz"
        path.write_bytes(path.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "SHA"):
            _verify_saved_artifacts(output, ["val_image_0"], bank)

    def test_formal_aggregate_mismatch_rejected(self):
        result, _calls, bank, output, _ = self._integration()
        modified = dict(result["confusion"])
        modified["a1_pixel"] = np.zeros((4, 4), dtype=int).tolist()
        with self.assertRaisesRegex(ValueError, "Whole-split"):
            _verify_saved_artifacts(output, ["val_image_0"], bank, modified)

    def test_failure_preserves_partial_image_count_and_no_claim(self):
        summary = dict(image_count=17)
        try:
            raise IndexError("failed current pooling")
        except IndexError as error:
            _record_failure(self.root, summary, "front_same_pass", "image_18", error)
        self.assertEqual(summary["image_count"], 17)
        self.assertFalse(summary["execution_complete"])
        self.assertFalse(summary["full_validation_evidence"])
        self.assertIn("failed current pooling", (self.root / "traceback.txt").read_text())


if __name__ == "__main__":
    unittest.main()
