"""CPU contracts for explicit intermediate-checkpoint Test evaluation.

Uses the released fix2 checkpoint writer and actual four-head model, but no
BRIGHT images, server weights, Test878 data, CUDA, or optimizer updates.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_test as evaluator


SOURCE = Path(__file__).resolve().parent.parent / "soft_memory_q_rebased_v1_fix2"
TOOL = evaluator.support._load_tool(SOURCE)
from test_train import FakeRuntime  # Released CPU fixture; real models below.


class CheckpointEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.heads, cls.initial = TOOL.train._new_heads(SimpleNamespace(chunk_size=2), "cpu")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.train = self.root / "train"
        self.folder = self.train / "checkpoints"
        self.folder.mkdir(parents=True)
        source = self.root / "source.json"
        source.write_text('{"synthetic":true}\n')
        self.config = dict(mode="train", smoke=False, steps=5000, seed=3407,
                           topk=64, chunk_size=2, key_name="t128", workers=0,
                           lr=1e-4, weight_decay=.005, baseline_policy="independent",
                           baseline_root=None, baseline_evidence=None,
                           allow_smoke_values=False, output=str(self.train),
                           bank_root=str(self.root / "parent"),
                           values=str(self.root / "values"), protocol=str(source))
        self.protocol = dict(source_protocol=str(source),
                             source_protocol_sha256=evaluator.digest(source))
        code = TOOL.common.source_hash()
        self.manifest = dict(format=TOOL.common.FORMAT, source_hash=code,
                             arguments=self.config, protocol_info=self.protocol,
                             train_names=["train_%05d" % i for i in range(2918)],
                             val_names=["val_%05d" % i for i in range(450)])
        evaluator.write_json(self.train / "manifest.json", self.manifest)
        TOOL.train._checkpoint(self.folder, "checkpoint_002500.pt", self.heads, {},
                               2500, SimpleNamespace(**self.config), self.protocol,
                               self.initial, code, {str(source): evaluator.digest(source)})
        self.path = self.folder / "checkpoint_002500.pt"
        self.sidecar = self.path.with_suffix(".pt.json")

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, step=2500):
        return evaluator.validate_checkpoint(self.root, step, TOOL)

    def rewrite(self, change, sidecar_change=None):
        saved = torch.load(self.path, map_location="cpu", weights_only=False)
        change(saved)
        torch.save(saved, self.path)
        sidecar = json.loads(self.sidecar.read_text())
        sidecar["sha256"] = evaluator.digest(self.path)
        if sidecar_change is not None:
            sidecar_change(sidecar)
        evaluator.write_json(self.sidecar, sidecar)

    def test_original_writer_checkpoint_2500_accepted_when_summary_advanced(self):
        evaluator.write_json(self.train / "summary.json", dict(
            complete=False, phase="training", updates_per_arm={a: 3500 for a in evaluator.HEADS},
            last_checkpoint="checkpoint_003500.pt"))
        before = {p: p.read_bytes() for p in (self.path, self.sidecar, self.train / "summary.json")}
        formal = self.validate()
        self.assertEqual(formal.step, 2500)
        self.assertFalse(formal.saved["training_complete"])
        self.assertEqual(formal.saved["updates_per_arm"], {a: 2500 for a in evaluator.HEADS})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_later_failed_summary_does_not_rewrite_saved_checkpoint(self):
        evaluator.write_json(self.train / "summary.json", dict(
            complete=False, phase="failed", error="later process failed"))
        self.assertEqual(self.validate().step, 2500)

    def test_missing_requested_step_has_no_latest_fallback(self):
        with self.assertRaisesRegex(ValueError, "no fallback"):
            self.validate(2000)

    def test_appended_checkpoint_bytes_fail_sha(self):
        with self.path.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "sidecar, hash"):
            self.validate()

    def test_smoke_checkpoint_rejected_even_with_new_sha(self):
        self.rewrite(lambda s: s.update(smoke=True, mode="smoke"),
                     lambda s: s.update(smoke=True, mode="smoke"))
        with self.assertRaisesRegex(ValueError, "sidecar, hash"):
            self.validate()

    def test_unequal_arm_budget_rejected(self):
        self.rewrite(lambda s: s["updates_per_arm"].update(d=2499))
        with self.assertRaisesRegex(ValueError, "Serialized checkpoint"):
            self.validate()

    def test_tensor_change_rejected_after_outer_sha_recomputed(self):
        def mutate(saved):
            next(iter(saved["heads"]["d"].values())).reshape(-1)[0] += 1
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, "tensor hash"):
            self.validate()

    def test_config_mismatch_with_original_manifest_rejected(self):
        self.rewrite(lambda s: s["config"].update(topk=8))
        with self.assertRaisesRegex(ValueError, "Serialized checkpoint"):
            self.validate()

    def test_consistent_but_smoke_values_configuration_rejected(self):
        self.manifest["arguments"]["allow_smoke_values"] = True
        evaluator.write_json(self.train / "manifest.json", self.manifest)
        self.rewrite(lambda s: s["config"].update(allow_smoke_values=True))
        with self.assertRaisesRegex(ValueError, "full independent"):
            self.validate()

    def test_source_file_changed_rejected(self):
        Path(self.config["protocol"]).write_text('{"synthetic":false}\n')
        with self.assertRaisesRegex(ValueError, "protocol path/hash"):
            self.validate()


class ControlledRuntime(FakeRuntime):
    def prepare(self, image):
        example = super().prepare(image)
        base = torch.full((1, 4, 2, 2), .1)
        classes = torch.tensor([[[0, 1], [1, 3]]])
        base.scatter_(1, classes[:, None], .7)
        example.common["base_probability_full"] = base
        example.target = (torch.tensor([[[0, 1], [2, 255]]]) if image.endswith("0")
                          else torch.tensor([[[3, 2], [1, 0]]]))
        return example


class InferenceAndMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.names = ["synthetic_0", "synthetic_1"]
        self.runtime = ControlledRuntime(SimpleNamespace(), self.output)
        self.heads, _ = TOOL.train._new_heads(SimpleNamespace(chunk_size=2), "cpu")
        with torch.no_grad():
            self.heads["c"].joint_head[-1].bias[2] = 10
            self.heads["d"].joint_head[-1].bias[3] = 10
        for h in self.heads.values():
            h.requires_grad_(False).eval()

    def tearDown(self):
        self.tmp.cleanup()

    def infer(self):
        with redirect_stdout(io.StringIO()):
            return evaluator.infer(self.runtime, self.heads, self.names, TOOL, self.output, 3407)

    def test_actual_four_heads_infer_without_optimizer_or_weight_changes(self):
        hashes = {a: TOOL.common.module_hash(h) for a, h in self.heads.items()}
        with patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer during Test")):
            cm, transitions, events = self.infer()
        self.assertEqual(hashes, {a: TOOL.common.module_hash(h) for a, h in self.heads.items()})
        self.assertTrue(all(p.grad is None for h in self.heads.values() for p in h.parameters()))
        self.assertEqual(self.runtime.prepare_count, 2)  # shared forward, not 4 x images
        self.assertFalse((self.output / "predictions").exists())
        self.assertFalse(list(self.output.glob("*.npz")))
        # Independent hand-counted confusion: ignored255 does not contribute.
        baseline = np.array([[1,0,0,1],[0,2,0,0],[0,2,0,0],[1,0,0,0]])
        np.testing.assert_array_equal(cm["grid_probq"], baseline)
        np.testing.assert_array_equal(cm["a"], baseline)
        np.testing.assert_array_equal(cm["b"], baseline)
        expected_c = np.zeros((4,4), dtype=np.int64)
        expected_c[:,2] = [2,2,2,1]
        np.testing.assert_array_equal(cm["c"], expected_c)
        expected_d = np.zeros((4,4), dtype=np.int64)
        expected_d[:,3] = [2,2,2,1]
        np.testing.assert_array_equal(cm["d"], expected_d)
        report = evaluator.metric_result(cm, transitions, events, TOOL)
        self.assertAlmostEqual(report["metrics"]["grid_probq"]["foreground_miou"], 1/6)
        self.assertAlmostEqual(report["metrics"]["grid_probq"]["four_class_miou"], 5/24)
        self.assertEqual(report["transitions"]["c"]["rescued_by_gt_class"], [0,0,2,0])

    def test_saved_journal_transition_corruption_rejected(self):
        cm, changes, events = self.infer()
        path = self.output / "images.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["transitions"]["a"][0][0][0] += 1
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "Transition marginal"):
            evaluator.reopen_journal(path, self.names, cm, changes, events, TOOL)

    def test_truncated_journal_rejected(self):
        cm, changes, events = self.infer()
        path = self.output / "images.jsonl"
        path.write_text(path.read_text().splitlines()[0]+"\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluator.reopen_journal(path, self.names, cm, changes, events, TOOL)

    def test_duplicate_or_reordered_journal_rejected(self):
        cm, changes, events = self.infer()
        path = self.output / "images.jsonl"
        rows = path.read_text().splitlines()
        path.write_text(rows[0]+"\n"+rows[0]+"\n")
        with self.assertRaisesRegex(ValueError, "identity/order"):
            evaluator.reopen_journal(path, self.names, cm, changes, events, TOOL)

    def test_early_loader_end_does_not_report_completion(self):
        self.runtime.batches = lambda names, train, seed: iter(names[:1])
        with self.assertRaisesRegex(ValueError, "ended early"):
            self.infer()

    def test_invalid_count_types_or_negative_values_rejected(self):
        with self.assertRaisesRegex(ValueError, "count array"):
            evaluator.counts(np.zeros((4,4), dtype=float), (4,4))
        bad = np.zeros((4,4), dtype=np.int64)
        bad[0,0] = -1
        with self.assertRaisesRegex(ValueError, "count array"):
            evaluator.counts(bad, (4,4))


if __name__ == "__main__":
    unittest.main()
