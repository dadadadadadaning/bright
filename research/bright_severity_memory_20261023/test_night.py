"""Launcher contracts: serial work, stop on failure, no hidden GPU job killing."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import night


class NightContracts(unittest.TestCase):
    def test_busy_gpu_is_rejected(self):
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "1234\n", "")):
            with self.assertRaisesRegex(RuntimeError, "1234"):
                night.gpu_idle()

    def test_unknown_gpu_occupancy_fails_closed(self):
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "driver unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Cannot check GPU"):
                night.gpu_idle()

    def test_all_smokes_precede_training_and_use_identical_budget(self):
        config = dict(source_run="/source", addon_dir="/addon", seed=3407, steps=2500,
                      variants=list(night.VARIANTS))
        stages = list(night.commands(config, Path("/new")))
        self.assertEqual([s.split("_", 1)[0] for s, _ in stages], ["smoke"]*4+["train"]*4)
        for name, command in stages:
            self.assertEqual(command[command.index("--seed")+1], "3407")
            self.assertEqual(command[command.index("--steps")+1], "2500")
            self.assertEqual(command[-1], "/new/" + name)
            self.assertNotIn("test", command)

    def test_failure_prevents_following_subprocess_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            root = parent / "night"
            root.mkdir()
            night.write_json(root / "config.json", dict(source_files={}, variants=["original"],
                steps=2500, stage_timeout=60))
            night.write_json(root / "night_summary.json", dict(phase="starting", complete=False,
                test_accessed=False, stages={}))
            commands = [("smoke_original", [sys.executable, "-c", "print('first failure', flush=True); raise SystemExit(3)"]),
                        ("train_original", [sys.executable, "-c", "raise AssertionError('must not start')"])]
            with (parent / ".severity_night.lock").open("a+") as lease:
                args = night.parser().parse_args(["worker", "--run", str(root), "--lease-fd", str(lease.fileno())])
                with patch.object(night, "gpu_idle"), patch.object(night, "identity", return_value={}), \
                     patch.object(night, "commands", return_value=commands), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "smoke_original failed"):
                        night.worker(args)
            summary = json.loads((root / "night_summary.json").read_text())
            self.assertEqual(summary["phase"], "failed")
            self.assertFalse(summary["complete"])
            self.assertFalse((root / "train_original.log").exists())
            self.assertIn("first failure", (root / "smoke_original.log").read_text())
            self.assertEqual(summary["stages"]["smoke_original"]["returncode"], 3)

    def test_direct_worker_without_inherited_lock_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            night.write_json(root / "config.json", {})
            night.write_json(root / "night_summary.json", {})
            args = night.parser().parse_args(["worker", "--run", str(root)])
            with self.assertRaisesRegex(ValueError, "inherited lease"):
                night.worker(args)

    def test_source_mutation_stops_before_gpu_child(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            root = parent / "run"
            root.mkdir()
            night.write_json(root / "config.json", dict(source_files={"file.py": "old"}, variants=["original"]))
            night.write_json(root / "night_summary.json", dict(complete=False, stages={}))
            with (parent / ".severity_night.lock").open("a+") as lease:
                args = night.parser().parse_args(["worker", "--run", str(root), "--lease-fd", str(lease.fileno())])
                with patch.object(night, "identity", return_value={"file.py": "new"}), \
                     patch.object(night, "commands", return_value=[("smoke_original", [])]), \
                     patch.object(subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        night.worker(args)
                    popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
