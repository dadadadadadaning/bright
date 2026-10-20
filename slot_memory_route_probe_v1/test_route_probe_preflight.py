"""Stdlib-only provenance and output-isolation tests for the route probe."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import route_probe
from inspect_failure import sha256


class RouteProbePreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "original"
        self.train = self.root / "seed_3407/train"
        self.train.mkdir(parents=True)
        self.replay = self.base / "failed_replay"
        (self.replay / "primary").mkdir(parents=True)
        self.arms = list(route_probe.ARMS)
        self.release = hashlib.sha256(b"synthetic addon release").hexdigest()
        self.plan = {
            "format": "BRIGHT_SLOT_MEMORY_ABLATION_V1",
            "arms": self.arms, "seeds": [3407],
            "addon_release_sha256": self.release,
            "source": {
                "config": {"repo": str(self.base / "repo")},
                "package": str(self.base / "source_package"),
                "source_run": str(self.base / "source_run"),
            },
        }
        self.manifest = {
            "format": self.plan["format"], "seed": 3407,
            "plan": copy.deepcopy(self.plan),
        }
        self.write_json(self.root / "plan.json", self.plan)
        self.write_json(self.train / "manifest.json", self.manifest)
        self.write_json(self.train / "summary.json", {
            "phase": "failed", "complete": False,
            "updates_per_arm": {a: 6 for a in self.arms},
        })
        self.rows = [
            {
                "step": step, "image": "synthetic_image_%d" % step,
                "epoch": 0, "epoch_seed": 3407, "event": "synthetic",
                "arms": {a: {"optimizer_step": step} for a in self.arms},
            } for step in range(1, 7)
        ]
        self.write_journal()
        self.parent = self.make_checkpoint(2)
        self.newer = self.make_checkpoint(5)
        self.failure = {
            "format": "BRIGHT_SLOT_MEMORY_FAILURE_REPLAY_V1",
            "phase": "worker_failed",
            "error": "ValueError: " + route_probe.ROUTE_ERROR,
            "lineage": {
                "run_root": str(self.root), "seed": 3407,
                "last_original_complete_step": 6, "parent_step": 2,
                "parent_checkpoint_sha256": sha256(self.parent),
                "original_journal_sha256": sha256(self.train / "train.jsonl"),
                "original_manifest_sha256": sha256(self.train / "manifest.json"),
            },
            "last_complete_step": 3,
            "last_marker": {
                "step": 4, "image": "synthetic_image_4",
                "arm": "b", "phase": "forward_begin", "completed_arms": ["e", "f"],
            },
        }
        self.failure_path = self.replay / "primary/worker_summary.json"
        self.write_json(self.failure_path, self.failure)
        self.trace = self.replay / "primary/traceback.txt"
        self.trace.write_text(
            "Synthetic traceback\n"
            "    parent_ids = validate_routes(q, ids, regions, 1, height * width)\n"
            "ValueError: " + route_probe.ROUTE_ERROR + "\n",
            encoding="utf-8",
        )
        self.bridge = SimpleNamespace(
            check_package=mock.Mock(return_value=self.release),
            verify_ledgers=mock.Mock(),
        )
        self.args = argparse.Namespace(
            replay_output=str(self.replay), addon_dir=None,
            output_base=str(self.base / "diagnostics"),
        )

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_journal(self):
        (self.train / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )

    def make_checkpoint(self, step):
        folder = self.train / "checkpoints"
        folder.mkdir(exist_ok=True)
        path = folder / ("checkpoint_%06d.pt" % step)
        path.write_bytes(b"Synthetic bytes only; never torch.load. " + str(step).encode())
        self.write_json(path.with_suffix(".pt.json"), {
            "step": step, "sha256": sha256(path), "roundtrip_verified": True,
            "head_hashes": {
                arm: hashlib.sha256(arm.encode()).hexdigest() for arm in self.arms
            },
        })
        return path

    def snapshot(self):
        return {
            str(path.relative_to(self.base)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.base.rglob("*") if path.is_file()
        }

    def preflight(self):
        with mock.patch.object(route_probe, "read_bridge", return_value=self.bridge):
            return route_probe.preflight(self.args)

    def test_valid_context_uses_bound_parent_instead_of_newer_checkpoint(self):
        before = self.snapshot()
        context = self.preflight()
        self.assertEqual(context["chosen"]["step"], 2)
        self.assertEqual(context["chosen"]["sha256"], sha256(self.parent))
        self.assertNotEqual(context["chosen"]["sha256"], sha256(self.newer))
        self.assertEqual(context["historical_step"], 4)
        self.assertEqual(context["image"], "synthetic_image_4")
        self.assertEqual(context["root"], self.root)
        self.assertEqual(context["replay"], self.replay)
        self.assertEqual(context["seed"], 3407)
        self.assertEqual(context["addon"], self.base / "repo/New/slot_memory_ablation_v1")
        self.assertEqual(len(context["evidence_files"]), 6)
        self.assertIn(str(self.parent), context["evidence_files"])
        self.assertNotIn(str(self.newer), context["evidence_files"])
        self.bridge.check_package.assert_called_once_with(context["addon"])
        self.bridge.verify_ledgers.assert_called_once_with(self.plan["source"])
        self.assertEqual(self.snapshot(), before)

    def test_changed_parent_bytes_are_rejected_even_with_updated_sidecar(self):
        self.parent.write_bytes(self.parent.read_bytes() + b"changed")
        sidecar = json.loads(self.parent.with_suffix(".pt.json").read_text())
        sidecar["sha256"] = sha256(self.parent)
        self.write_json(self.parent.with_suffix(".pt.json"), sidecar)
        with self.assertRaisesRegex(ValueError, "exact original parent checkpoint"):
            self.preflight()
        self.bridge.check_package.assert_not_called()

    def test_wrong_parent_hash_cannot_fall_back_to_newer_valid_checkpoint(self):
        self.failure["lineage"]["parent_checkpoint_sha256"] = sha256(self.newer)
        self.write_json(self.failure_path, self.failure)
        with self.assertRaisesRegex(ValueError, "exact original parent checkpoint"):
            self.preflight()

    def test_changed_journal_is_rejected_even_when_step_counts_still_match(self):
        self.rows[0]["image"] = "changed_image"
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "journal or manifest changed"):
            self.preflight()

    def test_changed_manifest_is_rejected_even_when_run_identity_still_matches(self):
        self.manifest["additional_field"] = "changed after replay"
        self.write_json(self.train / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "journal or manifest changed"):
            self.preflight()

    def test_failed_run_requires_complete_matching_six_arm_update_counts(self):
        for mutation in ("summary_count", "incomplete_tail", "lineage_count"):
            with self.subTest(mutation=mutation):
                self.write_journal()
                self.write_json(self.train / "summary.json", {
                    "phase": "failed", "complete": False,
                    "updates_per_arm": {a: 6 for a in self.arms},
                })
                changed_failure = copy.deepcopy(self.failure)
                if mutation == "summary_count":
                    self.write_json(self.train / "summary.json", {
                        "phase": "failed", "complete": False,
                        "updates_per_arm": {a: 5 for a in self.arms},
                    })
                elif mutation == "incomplete_tail":
                    with (self.train / "train.jsonl").open("a") as stream:
                        stream.write('{"step": 7')
                else:
                    changed_failure["lineage"]["last_original_complete_step"] = 5
                self.write_json(self.failure_path, changed_failure)
                with self.assertRaisesRegex(ValueError, "failed run/journal identity differs"):
                    self.preflight()

    def test_rejects_wrong_b_update_context(self):
        changes = (
            {"arm": "d"}, {"phase": "backward_begin"},
            {"completed_arms": ["e"]}, {"completed_arms": ["e", "f", "b"]},
            {"step": 2}, {"step": 7},
        )
        for changeset in changes:
            with self.subTest(changes=changeset):
                failure = copy.deepcopy(self.failure)
                failure["last_marker"].update(changeset)
                self.write_json(self.failure_path, failure)
                with self.assertRaisesRegex(ValueError, "shared parent-route failure on B"):
                    self.preflight()

    def test_rejects_unrelated_error_or_child_route_trace(self):
        failure = copy.deepcopy(self.failure)
        failure["error"] = "RuntimeError: CUDA error: device-side assert triggered"
        self.write_json(self.failure_path, failure)
        with self.assertRaisesRegex(ValueError, "failed route-validation replay worker"):
            self.preflight()
        self.write_json(self.failure_path, self.failure)
        self.trace.write_text(
            "child_ids = validate_routes(route_q, route_ids, 2 * regions, 1, height * width)\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "parent q validator"):
            self.preflight()

    def test_rejects_wrong_six_arm_order_before_bridge_load(self):
        self.plan["arms"] = ["f", "e", "b", "d", "r", "l"]
        self.manifest["plan"] = copy.deepcopy(self.plan)
        self.write_json(self.root / "plan.json", self.plan)
        self.write_json(self.train / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "original six-arm order"):
            self.preflight()
        self.bridge.check_package.assert_not_called()

    def test_bridge_release_and_source_ledger_failures_propagate(self):
        self.bridge.check_package.return_value = "wrong release"
        with self.assertRaisesRegex(ValueError, "Original addon differs"):
            self.preflight()
        self.bridge.check_package.return_value = self.release
        self.bridge.verify_ledgers.side_effect = ValueError("changed source ledger")
        with self.assertRaisesRegex(ValueError, "changed source ledger"):
            self.preflight()

    def test_output_rejects_protected_directory_and_descendants_without_writes(self):
        before = self.snapshot()
        for output in (self.root, self.root / "nested/diagnostics"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "outside original inputs"):
                    route_probe.new_output(output, [self.root])
        self.assertFalse((self.root / "nested").exists())
        self.assertEqual(self.snapshot(), before)

    def test_output_resolves_symlink_aliases_before_protection_check(self):
        alias = self.base / "original_alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside original inputs"):
            route_probe.new_output(alias / "nested", [self.root])
        with self.assertRaisesRegex(ValueError, "outside original inputs"):
            route_probe.new_output(self.root / "nested", [alias])
        self.assertFalse((self.root / "nested").exists())

    def test_output_creation_never_reuses_or_overwrites_an_existing_probe(self):
        output_base = self.base / "diagnostics"
        fixed = dt.datetime(2026, 10, 20, 1, 2, 3, tzinfo=dt.timezone.utc)
        with mock.patch.object(route_probe.dt, "datetime") as date:
            date.now.return_value = fixed
            with mock.patch.object(route_probe.os, "getpid", return_value=123):
                first = route_probe.new_output(output_base, [self.root, self.replay])
                sentinel = first / "keep.txt"
                sentinel.write_text("preserve", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    route_probe.new_output(output_base, [self.root, self.replay])
        self.assertEqual(first.parent, output_base)
        self.assertEqual(first.name, "route_probe_20261020_010203_123")
        self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
