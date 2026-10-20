"""Read-only preflight contracts using synthetic files and the real inspector."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import resume_context as context
from inspect_failure import sha256


class ResumeContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "original"
        self.train = self.root / "seed_3407/train"
        (self.train / "checkpoints").mkdir(parents=True)
        self.names = ["train_%04d" % index for index in range(2918)]
        self.val_names = ["val_%04d" % index for index in range(450)]
        source = dict(
            package=str(self.base / "source_package"), source_run=str(self.base / "source_run"),
            config=dict(repo=str(self.base / "repo"), bank_root=str(self.base / "banks"),
                        values=str(self.base / "values"), membership_root=str(self.base / "members")),
        )
        self.plan = dict(
            format="BRIGHT_SLOT_MEMORY_ABLATION_V1", arms=list(context.ARMS), seeds=[3407],
            endpoint_step=2500, fixed_train_count=32, source=source,
            addon_release_sha256="a" * 64,
        )
        self.manifest = dict(
            format=self.plan["format"], plan=copy.deepcopy(self.plan), seed=3407,
            train_names=self.names, val_names=self.val_names,
            fixed_train_names=sorted(random.Random(3598).sample(self.names, 32)),
            runtime_config={}, protocol_info={},
        )
        self.status = dict(
            phase="failed", complete=False,
            updates_per_arm={arm: 1396 for arm in context.ARMS},
        )
        self.write_original()
        self.original_checkpoint = self.checkpoint(self.train, 1000)
        self.bridge = SimpleNamespace(
            check_package=Mock(return_value="a" * 64), verify_ledgers=Mock(),
        )
        self.bridge_patch = patch.object(context, "read_bridge", return_value=self.bridge)
        self.bridge_patch.start()
        self.addCleanup(self.bridge_patch.stop)
        self.args = argparse.Namespace(
            original_run=str(self.root), seed=3407, addon_dir=None,
            parent_root=None, checkpoint_every=50, output_base=str(self.base / "outputs"),
        )

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def write_original(self):
        self.manifest["plan"] = copy.deepcopy(self.plan)
        self.write(self.root / "plan.json", self.plan)
        self.write(self.train / "manifest.json", self.manifest)
        self.write(self.train / "summary.json", self.status)
        with (self.train / "train.jsonl").open("w") as stream:
            for step in range(1, 1397):
                stream.write(json.dumps(dict(
                    step=step, image=self.names[step - 1], epoch=0, epoch_seed=3407,
                    arms={arm: {"optimizer_step": step} for arm in context.ARMS},
                )) + "\n")

    def checkpoint(self, root, step, lineage=None, valid=True):
        path = root / "checkpoints" / ("checkpoint_%06d.pt" % step)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(("synthetic checkpoint %s" % step).encode())
        sidecar = dict(
            step=step, sha256=sha256(path) if valid else "0" * 64, roundtrip_verified=True,
            head_hashes={arm: "b" * 64 for arm in context.ARMS},
        )
        if lineage is not None:
            sidecar.update(
                format=context.CHECKPOINT_FORMAT, lineage=lineage,
                updates_per_arm={arm: step for arm in context.ARMS},
            )
        self.write(path.with_suffix(".pt.json"), sidecar)
        return path

    def parent(self):
        baseline = context.prepare(self.args)
        root = self.base / "resume_parent"
        root.mkdir()
        self.write(root / "manifest.json", dict(
            format=context.FORMAT, lineage=baseline["lineage"],
            original_manifest=self.manifest,
        ))
        self.write(root / "summary.json", dict(format=context.FORMAT, complete=False))
        (root / "worker.lock").write_text("old worker pid\n")
        # A failed update may leave a truncated suffix; payload loading validates
        # the complete prefix through the selected saved step later.
        (root / "train.jsonl").write_text('{"partial":')
        self.checkpoint(root, 1050, baseline["lineage"])
        self.args.parent_root = str(root)
        return root, baseline

    def snapshot(self):
        return {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.base.rglob("*") if path.is_file()
        }

    def test_initial_start_uses_fixed_original1000_and_writes_nothing(self):
        self.checkpoint(self.train, 1200)
        before = self.snapshot()
        result = context.prepare(self.args)
        self.assertEqual(result["parent"]["kind"], "original")
        self.assertEqual(result["parent"]["step"], 1000)
        self.assertEqual(result["parent"]["path"], self.original_checkpoint)
        self.assertEqual(result["parent_journal"], self.train / "train.jsonl")
        self.assertEqual(result["checkpoint_every"], 50)
        self.assertEqual(set(result["lineage"]), {
            "original_run", "seed", "original_manifest_sha256", "original_journal_sha256",
            "original_parent_step", "original_parent_checkpoint_sha256", "endpoint_step",
        })
        self.assertEqual(self.snapshot(), before)
        self.bridge.verify_ledgers.assert_called_once_with(self.plan["source"])

    def test_full_unique_disjoint_splits_and_fixed_subset_are_required(self):
        cases = (
            lambda: self.manifest["train_names"].pop(),
            lambda: self.manifest["val_names"].__setitem__(0, self.names[0]),
            lambda: self.manifest["train_names"].__setitem__(1, self.names[0]),
            lambda: self.manifest["fixed_train_names"].reverse(),
        )
        for mutate in cases:
            with self.subTest(case=mutate):
                saved = copy.deepcopy(self.manifest)
                mutate()
                self.write(self.train / "manifest.json", self.manifest)
                with self.assertRaises(ValueError):
                    context.prepare(self.args)
                self.manifest = saved
                self.names = self.manifest["train_names"]
                self.val_names = self.manifest["val_names"]

    def test_endpoint_must_remain_in_first_epoch_after1000(self):
        for endpoint in (1000, 2919, True):
            with self.subTest(endpoint=endpoint):
                self.plan["endpoint_step"] = endpoint
                self.write_original()
                with self.assertRaises(ValueError):
                    context.prepare(self.args)

    def test_original_journal_must_be_complete_and_all_arms_match(self):
        with (self.train / "train.jsonl").open("a") as stream:
            stream.write('{"step":1397')
        with self.assertRaisesRegex(ValueError, "journal"):
            context.prepare(self.args)
        self.write_original()
        self.status["updates_per_arm"]["f"] = 1395
        self.write(self.train / "summary.json", self.status)
        with self.assertRaisesRegex(ValueError, "journal"):
            context.prepare(self.args)

    def test_original_checkpoint_bytes_and_addon_must_match(self):
        self.original_checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint1000"):
            context.prepare(self.args)
        self.checkpoint(self.train, 1000)
        self.bridge.check_package.return_value = "c" * 64
        with self.assertRaisesRegex(ValueError, "addon"):
            context.prepare(self.args)

    def test_newest_verified_formal_parent_allows_truncated_suffix(self):
        root, baseline = self.parent()
        self.checkpoint(root, 1100, baseline["lineage"])
        self.checkpoint(root, 1150, baseline["lineage"], valid=False)
        before = self.snapshot()
        result = context.prepare(self.args)
        self.assertEqual(result["lineage"], baseline["lineage"])
        self.assertEqual(result["parent"]["kind"], "resume")
        self.assertEqual(result["parent"]["step"], 1100)
        self.assertEqual(result["parent_journal"], root / "train.jsonl")
        self.assertEqual(result["parent_root"], root)
        self.assertEqual(self.snapshot(), before)
        self.assertIn(root, context.protected_paths(result))
        with context.inactive_parent(root):
            pass  # prepare has released its temporary parent lock.

    def test_diagnostic_completed_or_changed_parent_is_rejected(self):
        root, _ = self.parent()
        original = json.loads((root / "manifest.json").read_text())
        for field in ("format", "lineage", "original_manifest"):
            with self.subTest(field=field):
                altered = copy.deepcopy(original)
                altered[field] = "diagnostic_or_other_run"
                self.write(root / "manifest.json", altered)
                with self.assertRaises(ValueError):
                    context.prepare(self.args)
        self.write(root / "manifest.json", original)
        self.write(root / "summary.json", dict(complete=True))
        with self.assertRaisesRegex(ValueError, "already complete"):
            context.prepare(self.args)

    def test_diagnostic_checkpoint_or_wrong_lineage_is_rejected(self):
        root, _ = self.parent()
        sidecar = root / "checkpoints/checkpoint_001050.pt.json"
        original = json.loads(sidecar.read_text())
        for field, replacement in (
            ("format", "BRIGHT_SLOT_MEMORY_DIAGNOSTIC_CHECKPOINT_V1"),
            ("lineage", {"different": True}),
            ("updates_per_arm", {arm: 1049 for arm in context.ARMS}),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = replacement
                self.write(sidecar, changed)
                with self.assertRaisesRegex(ValueError, "No verified formal"):
                    context.prepare(self.args)

    def test_running_parent_is_rejected_and_lock_is_not_modified(self):
        root, _ = self.parent()
        lock = root / "worker.lock"
        before = (lock.read_bytes(), lock.stat().st_mtime_ns)
        descriptor = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "running or locked"):
                context.prepare(self.args)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns), before)
        with context.inactive_parent(root):
            with self.assertRaisesRegex(RuntimeError, "running or locked"):
                with context.inactive_parent(root):
                    pass
        with context.inactive_parent(root):
            pass

    def test_missing_parent_lock_is_not_created(self):
        root, _ = self.parent()
        (root / "worker.lock").unlink()
        with self.assertRaisesRegex(ValueError, "existing resume-parent worker lock"):
            context.prepare(self.args)
        self.assertFalse((root / "worker.lock").exists())

    def test_protected_paths_include_original_source_and_parent_inputs(self):
        result = context.prepare(self.args)
        expected = {self.root, result["addon"]}
        expected.update(Path(value) for value in self.plan["source"]["config"].values())
        expected.update(Path(self.plan["source"][key]) for key in ("package", "source_run"))
        self.assertTrue(expected <= set(context.protected_paths(result)))


if __name__ == "__main__":
    unittest.main()
