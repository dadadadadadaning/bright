"""Focused source-provenance tests without BRIGHT images or GPU weights."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

try:
    from . import fresh_source
except ImportError:
    import fresh_source


class TinyFront(nn.Module):
    def __init__(self, arm, cue_gain, width):
        super().__init__()
        if arm != "baseline" or cue_gain != 0.1:
            raise ValueError("Unexpected synthetic Q arm")
        self.base = nn.Linear(width, width)
        self.adapter = nn.Linear(width, width)

    def adapter_state_dict(self):
        return {"adapter." + name: value.detach().clone()
                for name, value in self.adapter.state_dict().items()}

    def load_adapter_state_dict(self, state):
        self.adapter.load_state_dict({name.removeprefix("adapter."): value
                                      for name, value in state.items()}, strict=True)


class FreshQSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.repo = root / "repo"
        (self.repo / "bda_benchmark").mkdir(parents=True)
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        self.train_list = root / "train_set.txt"
        self.val_list = root / "val_set.txt"
        self.train_list.write_text("".join(f"train_{i}.png\n" for i in range(2918)))
        self.val_list.write_text("".join(f"val_{i}.png\n" for i in range(450)))
        self.a1 = root / "a1.pth"
        self.a1.write_bytes(b"synthetic audited A1")
        self.e0 = root / "e0_protocol.json"
        self.e0.write_text(json.dumps(dict(
            format="SRAC_HARDQUERY_PROTOCOL_V1", scope="all",
            profile="legacy", numeric_overrides={},
            runtime_settings=dict(device="synthetic-cpu"),
            arguments=dict(dataset=str(self.dataset), seed=3407, workers=0, device="cpu"),
            input_paths=dict(train_list=str(self.train_list), val_list=str(self.val_list),
                             front=str(root / "DELETED_E0_FRONT.pth"),
                             bank=str(root / "DELETED_OLD_J_BANK.pt")),
        )))
        self.train_dir = root / "q" / "formal" / "train_baseline"
        self.val_dir = root / "q" / "formal" / "val_baseline"
        self.train_dir.mkdir(parents=True)
        self.val_dir.mkdir(parents=True)
        self.checkpoint = self.train_dir / "last.pth"
        self.output = root / "new_run"
        self.immutable = dict(e0_files=True, protocol=True, q_code=True,
                              runtime=True, frozen_downstream=True)
        self.payload = dict(
            format=fresh_source.Q_FORMAT, arm="baseline", smoke=False,
            test_accessed=False, run_complete=True, completed_steps=2000,
            step=2000, arguments=dict(steps=2000, eval_every=2000),
            full_validation_evidence=True, initial_a1_reproduction=True,
            final_a1_reproduction=dict(passed=True),
            a1_unchanged=True, downstream_unchanged=True,
            immutable=self.immutable, source_hash="q-code-hash",
            inherited_source_hash="front-code-hash",
            a1_sha256=fresh_source.digest(self.a1),
            train_sha256=fresh_source.digest(self.train_list),
            val_sha256=fresh_source.digest(self.val_list),
            protocol_sha256=fresh_source.digest(self.e0),
            front_sha256="a" * 64,
            front_checkpoint=str(root / "DELETED_E0_FRONT.pth"),
            e0_profile="legacy", e0_pool_mode="native",
            e0_runtime=dict(device="synthetic-cpu"),
            config=dict(width=2), cue_gain=0.1,
            valid_miou=0.62,
            adapter=TinyFront("baseline", 0.1, 2).adapter_state_dict(),
        )
        self.train_summary = dict(
            format=fresh_source.Q_FORMAT, arm="baseline", complete=True,
            smoke=False, test_accessed=False, steps=2000,
            formal_checkpoint="last.pth", full_validation_evidence=True,
            initial_a1_reproduction=True, a1_unchanged=True,
            downstream_unchanged=True, immutable=self.immutable,
            best_valid_miou=0.62)
        self.manifest = {name: self.payload[name] for name in (
            "format", "arm", "smoke", "test_accessed", "arguments", "source_hash",
            "inherited_source_hash", "a1_sha256", "train_sha256", "val_sha256",
            "protocol_sha256", "front_sha256", "e0_pool_mode", "e0_profile",
            "e0_runtime", "config", "cue_gain")}
        self.val_result = dict(format=fresh_source.Q_EVAL_FORMAT, complete=True,
                               image_count=450, arm="baseline")
        self.val_summary = dict(
            format=fresh_source.Q_EVAL_FORMAT, complete=True,
            full_validation_evidence=True, image_count=450,
            expected_image_count=450, checkpoint_steps=2000,
            arm="baseline", smoke=False, test_accessed=False,
            training=False, native_a1_replay_ok=True,
            reproduction=dict(passed=True), q_a1_reproduction=dict(passed=True),
            immutable=dict(q_code=True, q_model=True, q_a1=True),
        )
        self.deps = SimpleNamespace(
            q=SimpleNamespace(QVariantFront=TinyFront,
                              q_source_hash=lambda: "q-code-hash",
                              inherited_source_hash=lambda: "front-code-hash"),
            common=SimpleNamespace(load_a1_strict=lambda model, path: None),
            numeric=SimpleNamespace(configure_runtime=lambda profile, **kwargs: None,
                                    runtime_settings=lambda: dict(device="synthetic-cpu")),
        )

    def _write(self, independent=True):
        torch.save(self.payload, self.checkpoint)
        (self.train_dir / "summary.json").write_text(json.dumps(self.train_summary))
        (self.train_dir / "manifest.json").write_text(json.dumps(self.manifest))
        if independent:
            self.val_result["checkpoint_sha256"] = fresh_source.digest(self.checkpoint)
            (self.val_dir / "result.json").write_text(json.dumps(self.val_result))
            self.val_summary["checkpoint_sha256"] = fresh_source.digest(self.checkpoint)
            self.val_summary["result_sha256"] = fresh_source.digest(self.val_dir / "result.json")
            (self.val_dir / "summary.json").write_text(json.dumps(self.val_summary))
        source_path = self.output / "source_protocol.json"
        if source_path.exists():
            source_path.unlink()
        return fresh_source.create_source_protocol(
            self.checkpoint, self.a1, self.e0, self.output, self.repo, "cpu")

    def _load(self):
        protocol = self._write()
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            return fresh_source.load_q_source(protocol, self.output)

    def test_reconstructs_frozen_q_without_deleted_e0_front_or_j_bank(self):
        bundle = self._load()
        self.assertFalse(Path(self.payload["front_checkpoint"]).exists())
        self.assertFalse(any(p.requires_grad for p in bundle["front"].parameters()))
        self.assertEqual(bundle["source_kind"], "completed_q_baseline")
        self.assertEqual(bundle["original_front_sha256"], "a" * 64)
        self.assertFalse(bundle["source_provenance"]["historical_e0_source_reverified"])
        self.assertEqual(bundle["args"].dataset, str(self.dataset))
        self.assertEqual(bundle["args"].pool_mode, "native")
        self.assertEqual(len(bundle["train_names"]), 2918)
        self.assertEqual(len(bundle["val_names"]), 450)
        self.assertEqual(set(bundle["input_paths"]), set(bundle["input_fingerprints"]))
        self.assertNotIn("front", bundle["input_paths"])
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            second = fresh_source.load_q_source(self.output / "source_protocol.json",
                                                self.output / "another_stage")
        self.assertEqual(bundle["source_provenance"], second["source_provenance"])
        self.assertEqual(bundle["input_fingerprints"], second["input_fingerprints"])

    def test_rejects_incomplete_independent_val(self):
        self.val_summary["full_validation_evidence"] = False
        with self.assertRaisesRegex(ValueError, "Val450"):
            self._load()

    def test_rejects_mismatched_q_code_hash(self):
        self.payload["source_hash"] = "changed-code"
        self.manifest["source_hash"] = "changed-code"
        with self.assertRaisesRegex(ValueError, "source code differs"):
            self._load()

    def test_rejects_incomplete_adapter(self):
        del self.payload["adapter"]["adapter.bias"]
        with self.assertRaisesRegex(ValueError, "parameter set is incomplete"):
            self._load()

    def test_rejects_runtime_drift_before_constructing_q_front(self):
        self.deps.numeric.runtime_settings = lambda: dict(device="different")
        with self.assertRaisesRegex(ValueError, "Current runtime differs"):
            self._load()

    def test_rejects_changed_split_even_when_old_e0_files_are_not_checked(self):
        self._write()
        self.train_list.write_text("".join(f"new_train_{i}.png\n" for i in range(2918)))
        protocol = self.output / "source_protocol.json"
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            with self.assertRaisesRegex(ValueError, "split differs"):
                fresh_source.load_q_source(protocol, self.output)

    def test_protocol_retry_reuses_only_identical_source_identity(self):
        protocol = self._write()
        before = fresh_source.digest(protocol)
        same = fresh_source.create_source_protocol(
            self.checkpoint, self.a1, self.e0, self.output, self.repo, "cpu")
        self.assertEqual(same, protocol)
        self.assertEqual(fresh_source.digest(protocol), before)
        self.a1.write_bytes(b"different A1")
        with self.assertRaisesRegex(ValueError, "Existing Q source protocol differs"):
            fresh_source.create_source_protocol(
                self.checkpoint, self.a1, self.e0, self.output, self.repo, "cpu")

    def _training_val_journal(self, image_count=450):
        a1 = dict(valid_miou=0.6, iou=[0.5, 0.6, 0.7, 0.8],
                  precision=[0.5, 0.6, 0.7, 0.8],
                  recall=[0.5, 0.6, 0.7, 0.8], f1=[0.5, 0.6, 0.7, 0.8])
        front = dict(valid_miou=0.62)
        checks = []
        for field in ("valid_miou", "iou", "precision", "recall", "f1"):
            values = [a1[field]] if field == "valid_miou" else a1[field]
            for index, actual in enumerate(values):
                checks.append(dict(name=f"a1_pixel/{field}/{index}", actual=actual,
                                   expected=actual, absolute_error=0.0, passed=True))
        self.payload["final_a1_reproduction"] = dict(
            passed=True, image_count=450, tolerance=1e-5,
            scope="a1_only_not_a_joint_training_gate", checks=checks)
        row = dict(step=2000, image_count=image_count, test_accessed=False,
                   metrics=dict(final=front, front_region=front, a1_pixel=a1))
        (self.train_dir / "validation.jsonl").write_text(json.dumps(row) + "\n")

    def test_training_final_val450_fallback_without_independent_report(self):
        self._training_val_journal()
        protocol = self._write(independent=False)
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            bundle = fresh_source.load_q_source(protocol, self.output)
        self.assertEqual(bundle["source_provenance"]["evidence_level"],
                         fresh_source.TRAINING_EVIDENCE)
        self.assertFalse(bundle["source_provenance"]["independent_validation_completed"])
        self.assertIn("q_train_validation", bundle["input_paths"])
        self.assertNotIn("q_val_summary", bundle["input_paths"])
        # A later independent report cannot silently upgrade this bank's source.
        self.val_summary["complete"] = True
        (self.val_dir / "summary.json").write_text(json.dumps(self.val_summary))
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            later = fresh_source.load_q_source(protocol, self.output / "later_stage")
        self.assertEqual(bundle["source_provenance"], later["source_provenance"])

    def test_training_fallback_rejects_partial_val(self):
        self._training_val_journal(image_count=449)
        protocol = self._write(independent=False)
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            with self.assertRaisesRegex(ValueError, "final-step Val450"):
                fresh_source.load_q_source(protocol, self.output)

    def test_in_progress_independent_summary_does_not_block_training_fallback(self):
        self._training_val_journal()
        (self.val_dir / "summary.json").write_text('{"complete": tru')
        protocol = self._write(independent=False)
        with patch.object(fresh_source, "_dependencies", return_value=self.deps):
            bundle = fresh_source.load_q_source(protocol, self.output)
        self.assertEqual(bundle["source_provenance"]["evidence_level"],
                         fresh_source.TRAINING_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
