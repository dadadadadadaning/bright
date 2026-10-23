"""Stdlib-only contracts for the new plain-replay and initial-Val evidence gate.

The prior observed-replay gate is deliberately stubbed: its own contracts
already exercise the 375-index coverage evidence. These tests check the new
requirements and never deserialize checkpoints or access CUDA.
"""
import builtins
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evidence
import preflight


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class PreflightContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.package = root / "package"
        self.parent = root / "night" / "train_original"
        self.observed = root / "backward158_observed"
        self.plain = root / "backward158_plain"
        for folder in (self.package, self.parent, self.observed, self.plain):
            folder.mkdir(parents=True)
        (self.parent / "checkpoints").mkdir()
        self.checkpoint = self.parent / "checkpoints" / "checkpoint_000150.pt"
        self.checkpoint.write_bytes(b"Not a pickle: stdlib preflight must not deserialize this.")
        train_names = [f"train_{index:05d}" for index in range(2918)]
        val_names = [f"val_{index:05d}" for index in range(450)]
        fixed = sorted(random.Random(3407 + 191).sample(train_names, 450))
        self.plan = dict(seed=3407, fixed_train_count=450)
        self.manifest = dict(seed=3407, train_names=train_names, val_names=val_names,
            fixed_train_names=fixed, protocol_info={"immutable_source": "fixture"},
            runtime_config=dict(output=str(self.parent), seed=3407, device="cuda:0", workers=1))
        self.parent_rows = [dict(step=step, image=train_names[step-1]) for step in range(1, 158)]
        self.parent_hashes = {str(self.checkpoint): evidence.sha(self.checkpoint)}
        baseline_rows = [dict(step=step, optimizer_step=step, image=f"sample_{step}",
            epoch=0, variant="original", lr=step * 2e-7,
            original_loss=0.2 + step / 1000, training_loss=0.2 + step / 1000,
            grad_norm_preclip=0.3 + step / 1000) for step in range(151, 159)]
        self.observed_info = dict(root=str(self.observed), rows=baseline_rows,
                                  fingerprints={str(self.observed / "verified.json"): "recorded-hash"})
        self.parent_result = (SimpleNamespace(), self.plan, self.manifest, self.parent_rows,
                              self.checkpoint, self.parent_hashes, self.observed_info)
        self.rows = copy.deepcopy(baseline_rows)
        for row in self.rows:
            row["original_loss"] += 1e-7
            row["training_loss"] = row["original_loss"]
        self.runner = self.plain / "runner.pyz"
        self.runner.write_bytes(b"Plain replay fixture artifact")
        self.runner_sha = evidence.sha(self.runner)
        summary = dict(format="BRIGHT_BACKWARD158_PLAIN_V1", phase="plain_window_complete",
            complete=False, diagnostic_complete=True, training_complete=False,
            full_validation_evidence=False, full_test_evidence=False, test_accessed=False,
            optimizer_updates_returned=8, completed_update_calls=8, new_complete_updates=8,
            restored_step=150, last_complete_step=158, bounded_endpoint=158,
            original_endpoint=2500, optimizer_state_restored=True,
            backward_index_observer=False, anomaly_detection=False, cuda_launch_blocking="1",
            failed_step_optimizer_execution_unknown=False, saved_tensors_hooks_used=False,
            tensor_values_repaired=False, failed_samples_skipped=False, automatic_retry=False,
            original_failed_runs_unchanged=True, observed_baseline_unchanged=True,
            historical_failure_resolved=False, exact_historical_replay=False,
            error=None, parent=str(self.parent), baseline=str(self.observed),
            runtime_integrity={"source_files": True, "frozen_front": True},
            pool_guard=dict(restored=True, failures=[], restore_conflicts=[]))
        dump(self.plain / "summary.json", summary)
        dump(self.plain / "supervisor.json", dict(job_finished=True, returncode=0,
            phase="plain_window_complete", error=None))
        self.replay_manifest = dict(parent_manifest=self.manifest,
            protocol_info=self.manifest["protocol_info"], rng_restored=True,
            optimizer_state_restored=True, data_order_prefix_verified=True,
            runtime_config={**self.manifest["runtime_config"], "output": str(self.plain)})
        dump(self.plain / "manifest.json", self.replay_manifest)
        dump(self.plain / "lineage.json", dict(parent=str(self.parent), checkpoint=str(self.checkpoint),
            checkpoint_sha256=evidence.sha(self.checkpoint), parent_fingerprints=self.parent_hashes,
            baseline=str(self.observed), baseline_fingerprints=self.observed_info["fingerprints"],
            steps=list(range(151, 159)), formal_checkpoint_written=False, test_accessed=False))
        self.write_updates()
        (self.parent / "initial_val").mkdir()
        self.initial_journal = self.parent / "initial_val" / "images.jsonl"
        self.initial_journal.write_text("".join(json.dumps(dict(index=index, image=name)) + "\n"
                                               for index, name in enumerate(val_names)))
        self.initial_path = self.parent / "initial_val" / "result.json"
        dump(self.initial_path, dict(step=0, split="val", variant="original", image_count=450,
            requested_names=val_names, execution_complete=True, journal_verified=True,
            test_accessed=False, images_sha256=evidence.sha(self.initial_journal),
            phase="evaluation_complete_integrity_pending", complete=False,
            full_validation_evidence=False, final_integrity_pending=True))

    def write_updates(self):
        (self.plain / "diagnostic_updates.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows))
        dump(self.plain / "comparison.json", evidence.compare_rows(self.observed_info["rows"], self.rows))

    def mutate(self, path, **updates):
        content = evidence.read(path)
        content.update(updates)
        dump(path, content)

    def check(self):
        with patch.object(evidence, "check_pair", return_value=self.parent_result) as old_gate, \
             patch.object(preflight, "PLAIN_SHA256", self.runner_sha):
            result = preflight.check_inputs(self.package, self.parent, self.observed, self.plain)
        old_gate.assert_called_once_with(self.package, self.parent, self.observed)
        return result

    def test_accepts_complete_plain_and_returns_bound_file_fingerprints(self):
        result = self.check()
        self.assertEqual(len(result), 8)
        self.assertEqual(result[:7], self.parent_result)
        info = result[-1]
        self.assertEqual(info["rows"], self.rows)
        self.assertEqual(info["root"], str(self.plain))
        self.assertEqual(len(info["fingerprints"]), 9)
        self.assertEqual(info["fingerprints"][str(self.runner)], self.runner_sha)

    def test_pending_initial_val_remains_unpromoted_and_files_are_unchanged(self):
        before = {str(path): evidence.sha(path) for path in self.temporary_paths()}
        initial = self.check()[-1]["initial_val"]
        self.assertEqual(initial["phase"], "evaluation_complete_integrity_pending")
        self.assertIs(initial["complete"], False)
        self.assertIs(initial["full_validation_evidence"], False)
        self.assertIs(initial["inherited_reference_only"], True)
        self.assertIs(initial["promoted_by_recovery"], False)
        self.assertEqual(before, {str(path): evidence.sha(path) for path in self.temporary_paths()})
        self.assertIs(evidence.read(self.initial_path)["final_integrity_pending"], True)

    def temporary_paths(self):
        return sorted(path for path in Path(self.temporary.name).rglob("*") if path.is_file())

    def test_incomplete_plain_is_rejected(self):
        self.mutate(self.plain / "summary.json", diagnostic_complete=False)
        with self.assertRaisesRegex(ValueError, "diagnostic_complete"):
            self.check()

    def test_unfinished_supervisor_is_rejected(self):
        self.mutate(self.plain / "supervisor.json", job_finished=False)
        with self.assertRaisesRegex(ValueError, "supervisor"):
            self.check()

    def test_changed_plain_runner_is_rejected(self):
        self.runner.write_bytes(b"changed runner")
        with self.assertRaisesRegex(ValueError, "runner SHA"):
            self.check()

    def test_changed_lineage_checkpoint_is_rejected(self):
        self.mutate(self.plain / "lineage.json", checkpoint_sha256="different")
        with self.assertRaisesRegex(ValueError, "lineage inputs"):
            self.check()

    def test_changed_observed_fingerprints_are_rejected(self):
        self.mutate(self.plain / "lineage.json", baseline_fingerprints={"different": "hash"})
        with self.assertRaisesRegex(ValueError, "lineage inputs"):
            self.check()

    def test_changed_runtime_settings_are_rejected(self):
        self.mutate(self.plain / "manifest.json", runtime_config={
            **self.replay_manifest["runtime_config"], "workers": 2})
        with self.assertRaisesRegex(ValueError, "runtime config"):
            self.check()

    def test_comparison_that_does_not_match_journal_is_rejected(self):
        comparison_path = self.plain / "comparison.json"
        value = evidence.read(comparison_path)
        value["pairs"][0]["original_loss"]["absolute_difference"] += 0.1
        dump(comparison_path, value)
        with self.assertRaisesRegex(ValueError, "comparison differs"):
            self.check()

    def test_partial_plain_journal_is_rejected(self):
        self.rows.pop()
        self.write_updates()
        with self.assertRaisesRegex(ValueError, "journal incomplete"):
            self.check()

    def test_wrong_plain_sample_is_rejected_even_with_recomputed_scalar_summary(self):
        self.rows[0]["image"] = "different_sample"
        (self.plain / "diagnostic_updates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows))
        with self.assertRaisesRegex(ValueError, "sample/step mismatch"):
            self.check()

    def test_initial_val_journal_hash_mutation_is_rejected(self):
        self.initial_journal.write_text(self.initial_journal.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "initial Val evidence"):
            self.check()

    def test_initial_val_missing_image_rejected_even_if_hash_is_refreshed(self):
        lines = self.initial_journal.read_text().splitlines()
        self.initial_journal.write_text("\n".join(lines[:-1]) + "\n")
        self.mutate(self.initial_path, images_sha256=evidence.sha(self.initial_journal))
        with self.assertRaisesRegex(ValueError, "Val coverage"):
            self.check()

    def test_initial_val_wrong_requested_names_is_rejected(self):
        self.mutate(self.initial_path, requested_names=self.manifest["val_names"][:-1])
        with self.assertRaisesRegex(ValueError, "initial Val evidence"):
            self.check()

    def test_initial_val_wrong_order_rejected_even_if_hash_is_refreshed(self):
        rows = [json.loads(line) for line in self.initial_journal.read_text().splitlines()]
        rows[1]["index"] = 0
        self.initial_journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.mutate(self.initial_path, images_sha256=evidence.sha(self.initial_journal))
        with self.assertRaisesRegex(ValueError, "journal order"):
            self.check()

    def test_fixed_train_subset_must_match_original_seeded_recipe(self):
        fixed = self.manifest["fixed_train_names"]
        replacement = next(name for name in self.manifest["train_names"] if name not in fixed)
        fixed[0] = replacement
        dump(self.plain / "manifest.json", self.replay_manifest)
        with self.assertRaisesRegex(ValueError, "Fixed Train subset"):
            self.check()

    def test_stdlib_preflight_does_not_import_torch_or_deserialize_checkpoint(self):
        original_import = builtins.__import__

        def forbid_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Stdlib preflight imported Torch")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=forbid_torch), \
             patch("pickle.load", side_effect=AssertionError("Checkpoint was deserialized")), \
             patch("pickle.loads", side_effect=AssertionError("Checkpoint was deserialized")):
            self.check()
        self.assertTrue(self.checkpoint.read_bytes().startswith(b"Not a pickle"))


if __name__ == "__main__":
    unittest.main()
