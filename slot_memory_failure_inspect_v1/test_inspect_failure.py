"""Failure evidence contracts using synthetic files; never imports Torch."""
from __future__ import annotations

import copy
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from inspect_failure import checkpoint_info, main, read_journal, validate_identity


FORMAT = "BRIGHT_SLOT_MEMORY_ABLATION_V1"
ARMS = ("e", "f", "b", "d", "r", "l")
SEED = 3407


def record(step):
    return {
        "step": step,
        "epoch": 0,
        "epoch_seed": SEED,
        "image": "synthetic_%06d" % step,
        "event": "synthetic",
        "arms": {arm: {"optimizer_step": step} for arm in ARMS},
    }


class SyntheticRun(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / ("seed_%d" % SEED) / "train"
        self.train.mkdir(parents=True)

    def json_file(self, path, data):
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    def identity(self):
        plan = {
            "format": FORMAT,
            "arms": list(ARMS),
            "seeds": [SEED],
            "source": {"source_run": "/synthetic/source"},
        }
        manifest = {"format": FORMAT, "seed": SEED, "plan": copy.deepcopy(plan)}
        self.json_file(self.root / "plan.json", plan)
        self.json_file(self.train / "manifest.json", manifest)
        return plan, manifest

    def journal(self, records, suffix=""):
        path = self.train / "train.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in records) + suffix,
            encoding="utf-8",
        )
        return path


class IdentityTests(SyntheticRun):
    def test_valid_identity_reads_without_modifying_files(self):
        expected_plan, expected_manifest = self.identity()
        before = {
            str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.root.rglob("*") if p.is_file()
        }
        plan, manifest = validate_identity(self.root, SEED)
        self.assertEqual(plan, expected_plan)
        self.assertEqual(manifest, expected_manifest)
        after = {
            str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.root.rglob("*") if p.is_file()
        }
        self.assertEqual(before, after)

    def test_extended_experiment_format_is_rejected(self):
        plan, manifest = self.identity()
        plan["format"] = "BRIGHT_JOINT_ANCHOR_EXTENDED_V2"
        manifest["format"] = plan["format"]
        manifest["plan"] = plan
        self.json_file(self.root / "plan.json", plan)
        self.json_file(self.train / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            validate_identity(self.root, SEED)

    def test_manifest_identity_mismatch_is_rejected(self):
        for mutation in ("seed", "arms", "source"):
            with self.subTest(mutation=mutation):
                _, manifest = self.identity()
                if mutation == "seed":
                    manifest["seed"] = SEED + 1
                elif mutation == "arms":
                    manifest["plan"]["arms"] = list(ARMS[:-1])
                else:
                    manifest["plan"]["source"]["source_run"] = "/other/source"
                self.json_file(self.train / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    validate_identity(self.root, SEED)

    def test_seed_must_be_in_plan(self):
        plan, manifest = self.identity()
        plan["seeds"] = [SEED + 1]
        manifest["plan"] = plan
        self.json_file(self.root / "plan.json", plan)
        self.json_file(self.train / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            validate_identity(self.root, SEED)


class JournalTests(SyntheticRun):
    def test_contiguous_all_arm_updates_have_next_unjournaled_step(self):
        path = self.journal([record(1), record(2), record(3)])
        before = path.read_bytes()
        result = read_journal(path, ARMS)
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["last_complete"]["step"], 3)
        self.assertEqual(result["contiguous_prefix_step"], 3)
        self.assertEqual(result["next_unjournaled_step"], 4)
        self.assertFalse(result["issues"])
        self.assertFalse(result["trailing_incomplete"])
        self.assertEqual(path.read_bytes(), before)

    def test_truncated_tail_does_not_count_as_completed_update(self):
        path = self.journal([record(1), record(2)], '{"step": 3, "arms": {')
        result = read_journal(path, ARMS)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["last_complete"]["step"], 2)
        self.assertEqual(result["contiguous_prefix_step"], 2)
        self.assertTrue(result["trailing_incomplete"])
        self.assertEqual(result["next_unjournaled_step"], 3)

    def test_valid_json_without_record_terminator_is_still_uncommitted_tail(self):
        path = self.journal([record(1)], json.dumps(record(2)))
        result = read_journal(path, ARMS)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["last_complete"]["step"], 1)
        self.assertEqual(result["contiguous_prefix_step"], 1)
        self.assertTrue(result["trailing_incomplete"])
        self.assertEqual(result["next_unjournaled_step"], 2)

    def test_gap_or_duplicate_prevents_next_step_inference(self):
        for steps in ((1, 3), (1, 1), (2, 3)):
            with self.subTest(steps=steps):
                result = read_journal(self.journal([record(s) for s in steps]), ARMS)
                self.assertTrue(result["issues"])
                self.assertIsNone(result["next_unjournaled_step"])
                self.assertEqual(result["contiguous_prefix_step"], 1 if steps[0] == 1 else 0)

    def test_incomplete_or_mismatched_arm_update_is_not_a_complete_step(self):
        for mutation in ("missing", "wrong_count", "extra"):
            with self.subTest(mutation=mutation):
                bad = record(2)
                if mutation == "missing":
                    bad["arms"].pop("l")
                elif mutation == "wrong_count":
                    bad["arms"]["d"]["optimizer_step"] = 1
                else:
                    bad["arms"]["a"] = {"optimizer_step": 2}
                result = read_journal(self.journal([record(1), bad]), ARMS)
                self.assertTrue(result["issues"])
                self.assertEqual(result["contiguous_prefix_step"], 1)
                self.assertIsNone(result["next_unjournaled_step"])

    def test_malformed_interior_line_cannot_be_ignored(self):
        path = self.journal([record(1)], "not JSON\n" + json.dumps(record(2)) + "\n")
        result = read_journal(path, ARMS)
        self.assertTrue(result["issues"])
        self.assertFalse(result["trailing_incomplete"])
        self.assertEqual(result["contiguous_prefix_step"], 1)
        self.assertIsNone(result["next_unjournaled_step"])

    def test_malformed_terminated_tail_is_not_a_partial_write_exception(self):
        path = self.journal([record(1)], '{"step":\n')
        result = read_journal(path, ARMS)
        self.assertTrue(result["issues"])
        self.assertFalse(result["trailing_incomplete"])
        self.assertEqual(result["contiguous_prefix_step"], 1)
        self.assertIsNone(result["next_unjournaled_step"])


class CheckpointTests(SyntheticRun):
    def checkpoint(self, step=1000):
        folder = self.train / "checkpoints"
        folder.mkdir(exist_ok=True)
        path = folder / ("checkpoint_%06d.pt" % step)
        path.write_bytes(b"synthetic bytes; never deserialize as a model")
        sidecar = {
            "step": step,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "head_hashes": {arm: "0" * 64 for arm in ARMS},
            "roundtrip_verified": True,
        }
        self.json_file(path.with_suffix(".pt.json"), sidecar)
        return folder, path, sidecar

    def test_checkpoint_hash_and_sidecar_are_verified_without_loading(self):
        folder, path, sidecar = self.checkpoint()
        result = checkpoint_info(folder, ARMS)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["step"], 1000)
        self.assertEqual(result[0]["name"], path.name)
        self.assertEqual(result[0]["bytes"], path.stat().st_size)
        self.assertEqual(result[0]["sha256"], sidecar["sha256"])
        self.assertTrue(result[0]["sidecar_sha256_match"])
        self.assertTrue(result[0]["verified"])

    def test_tampered_checkpoint_is_unverified(self):
        folder, path, _ = self.checkpoint()
        path.write_bytes(path.read_bytes() + b"changed")
        result = checkpoint_info(folder, ARMS)[0]
        self.assertFalse(result["sidecar_sha256_match"])
        self.assertFalse(result["verified"])

    def test_missing_sidecar_is_not_verified(self):
        folder, path, _ = self.checkpoint()
        path.with_suffix(".pt.json").unlink()
        result = checkpoint_info(folder, ARMS)[0]
        self.assertIsNone(result["sidecar_sha256_match"])
        self.assertFalse(result["verified"])

    def test_hash_alone_does_not_verify_wrong_step_or_arm_identity(self):
        for mutation in ("step", "arms", "roundtrip"):
            with self.subTest(mutation=mutation):
                folder, path, sidecar = self.checkpoint()
                if mutation == "step":
                    sidecar["step"] = 500
                elif mutation == "arms":
                    sidecar["head_hashes"].pop("l")
                else:
                    sidecar["roundtrip_verified"] = False
                self.json_file(path.with_suffix(".pt.json"), sidecar)
                result = checkpoint_info(folder, ARMS)[0]
                self.assertTrue(result["sidecar_sha256_match"])
                self.assertFalse(result["verified"])

    def test_cli_reads_synthetic_failed_run_without_importing_torch_or_writing(self):
        self.identity()
        self.journal([record(1), record(2)])
        self.checkpoint(step=2)
        self.json_file(self.train / "summary.json", {
            "phase": "failed", "complete": False,
            "updates_per_arm": {a: 2 for a in ARMS},
            "error": "RuntimeError: synthetic failure",
        })
        self.json_file(self.root / "workflow_state.json", {
            "phase": "failed", "current_stage": "3407_train",
        })
        (self.train / "traceback.txt").write_text(
            "".join("synthetic traceback line %d\n" % i for i in range(80)),
            encoding="utf-8",
        )
        log_dir = self.root / "logs"
        log_dir.mkdir()
        (log_dir / "3407_train.log").write_text(
            "".join("RuntimeError: synthetic error %d " % i + "x" * 600 + "\n"
                    for i in range(50)), encoding="utf-8",
        )
        before = {
            str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.root.rglob("*") if p.is_file()
        }
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Collector must not import Torch")
            return real_import(name, *args, **kwargs)

        capture = io.StringIO()
        with mock.patch("sys.argv", ["inspect_failure.py", "--run-root", str(self.root)]):
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                with contextlib.redirect_stdout(capture):
                    code = main()
        self.assertEqual(code, 0)
        text = capture.getvalue()
        self.assertIn("BEGIN ABLATION FAILURE INSPECTION", text)
        self.assertIn("END ABLATION FAILURE INSPECTION", text)
        self.assertIn("latest_checkpoint_matching_sidecar_and_journal = 2", text)
        self.assertIn('"next_unjournaled_step": 3', text)
        trace = [line for line in text.splitlines() if line.startswith("synthetic traceback")]
        signals = [line for line in text.splitlines() if line.startswith("RuntimeError:")]
        self.assertEqual(len(trace), 35)
        self.assertEqual(len(signals), 6)
        self.assertTrue(all(len(line) <= 500 for line in signals))
        after = {
            str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.root.rglob("*") if p.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
