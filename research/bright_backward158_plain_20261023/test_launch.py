"""CPU-only launcher contracts; no Torch, data loading, training or GPU access."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import launch


class LauncherContracts(unittest.TestCase):
    def make_run(self, base):
        base = Path(base)
        parent = base / "night_3407" / "train_original"
        parent.mkdir(parents=True)
        root = base / "backward158_plain_case"
        root.mkdir()
        (root / "runner.pyz").write_bytes(b"diagnostic snapshot")
        launch.write_json(root / "config.json", dict(format=launch.FORMAT,
            parent=str(parent), package=str(base / "package"), baseline=str(base / "observed"), output=str(root),
            timeout_seconds=60, runner_sha256=launch.digest(root / "runner.pyz")))
        launch.write_json(root / "summary.json", dict(phase="diagnostic_starting",
            complete=False, diagnostic_complete=False, test_accessed=False))
        return parent, root

    def args(self, command, root, lease):
        return launch.parser().parse_args([command, "--output", str(root),
                                           "--lease-fd", str(lease.fileno())])

    def test_busy_or_unqueryable_gpu_is_rejected(self):
        for result, text in [(subprocess.CompletedProcess([], 0, "123\n", ""), "123"),
                             (subprocess.CompletedProcess([], 1, "", "driver"), "Cannot check")]:
            with self.subTest(result=result), patch.object(launch.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, text):
                    launch.gpu_idle()

    def test_parent_root_is_original_study_root(self):
        self.assertEqual(launch.study_root("/tmp/study/night/train_original"), Path("/tmp/study"))

    def test_worker_requires_inherited_correct_lease(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (Path(base) / ".severity_night.lock").touch()
            with self.assertRaisesRegex(ValueError, "inherited lease"):
                launch.validate_lease(None, base)
            with (Path(base) / "wrong.lock").open("a+") as wrong:
                with self.assertRaisesRegex(ValueError, "Wrong inherited"):
                    launch.validate_lease(wrong.fileno(), base)

    def test_self_runner_output_beats_latest_marker(self):
        with tempfile.TemporaryDirectory() as base:
            parent, root = self.make_run(base)
            other = Path(base) / "backward158_plain_newer"
            other.mkdir()
            launch.write_json(Path(base) / launch.MARKER, dict(output=str(other)))
            args = launch.parser().parse_args(["logs", "--parent", str(parent)])
            with patch.object(launch.sys, "argv", [str(root / "runner.pyz"), "logs"]):
                self.assertEqual(launch.select_output(args), root)
            with patch.object(launch.sys, "argv", ["/tmp/download.pyz", "logs"]):
                self.assertEqual(launch.select_output(args), other)

    def test_explicit_output_beats_latest_and_outside_latest_rejected(self):
        with tempfile.TemporaryDirectory() as base:
            parent, root = self.make_run(base)
            launch.write_json(Path(base) / launch.MARKER, dict(output="/tmp"))
            explicit = launch.parser().parse_args(["status", "--parent", str(parent), "--output", str(root)])
            self.assertEqual(launch.select_output(explicit), root)
            implicit = launch.parser().parse_args(["status", "--parent", str(parent)])
            with patch.object(launch.sys, "argv", ["/tmp/download.pyz"]):
                with self.assertRaisesRegex(ValueError, "outside"):
                    launch.select_output(implicit)

    def test_worker_delegates_only_selected_parent_package_output(self):
        with tempfile.TemporaryDirectory() as base:
            parent, root = self.make_run(base)
            with (Path(base) / ".severity_night.lock").open("a+") as lease:
                args = self.args("worker", root, lease)
                with patch.object(launch.replay, "run_worker") as run:
                    self.assertEqual(launch.worker(args), 0)
                run.assert_called_once_with(args)
                self.assertEqual(args.parent, parent)
                self.assertEqual(args.output, root)
                self.assertEqual(args.package, Path(base) / "package")
                self.assertEqual(args.baseline, Path(base) / "observed")

    def test_start_creates_new_output_without_reusing_existing(self):
        with tempfile.TemporaryDirectory() as base:
            parent, original = self.make_run(base)
            (original / "evidence.txt").write_text("preserve")
            args = launch.parser().parse_args(["start", "--parent", str(parent), "--package", base])
            child = MagicMock(pid=12345)

            def snapshot(path):
                Path(path).write_bytes(b"new snapshot")
                return launch.digest(path)

            def spawn_checked(*args, **kwargs):
                self.assertTrue(fcntl.fcntl(kwargs['stdout'].fileno(), fcntl.F_GETFL) & os.O_APPEND)
                return child

            with patch.object(launch.replay, "check_pair"), patch.object(launch, "gpu_idle"), \
                 patch.object(launch, "snapshot_runner", side_effect=snapshot), \
                 patch.object(launch.subprocess, "Popen", side_effect=spawn_checked) as spawn, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.start(args), 0)
            output = Path(launch.read_json(Path(base) / launch.MARKER)["output"])
            self.assertNotEqual(output, original)
            self.assertTrue(output.name.startswith("backward158_plain_"))
            self.assertEqual((original / "evidence.txt").read_text(), "preserve")
            self.assertEqual(spawn.call_count, 1)
            options = spawn.call_args.kwargs
            self.assertTrue(options["start_new_session"])
            self.assertEqual(len(options["pass_fds"]), 1)
            self.assertEqual(options["env"]["CUDA_LAUNCH_BLOCKING"], "1")

    def test_start_rejects_over_budget_or_reused_output(self):
        for flags, message in [(["--timeout-seconds", "1801"], "1..1800"),
                               (["--output", "/tmp/old"], "new backward158")]:
            with self.subTest(flags=flags), patch.object(launch.replay, "check_pair") as read:
                with self.assertRaisesRegex(ValueError, message):
                    launch.start(launch.parser().parse_args(["start", *flags]))
                read.assert_not_called()

    def test_failed_worker_stops_without_retry_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "first_failure.json").write_text('{"original":"assert"}')
            launch.write_json(root / "summary.json", dict(phase="failed", error="first CUDA assert",
                complete=False, diagnostic_complete=False, test_accessed=False))
            child = MagicMock(pid=23456)
            child.wait.return_value = 7
            child.poll.return_value = 7
            with (Path(base) / ".severity_night.lock").open("a+") as lease, \
                 patch.object(launch, "gpu_idle"), \
                 patch.object(launch.subprocess, "Popen", return_value=child) as spawn, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.supervise(self.args("supervise", root, lease)), 2)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(launch.read_json(root / "summary.json")["error"], "first CUDA assert")
            self.assertEqual(launch.read_json(root / "first_failure.json"), {"original": "assert"})
            status = launch.read_json(root / "supervisor.json")
            self.assertEqual(status["retry_count"], 0)
            self.assertTrue(status["job_finished"])
            self.assertFalse(status["complete"])

    def test_timeout_terminates_only_owned_group_and_writes_failed_summary(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "summary.json").unlink()
            child = MagicMock(pid=34567)
            child.poll.return_value = None
            with (Path(base) / ".severity_night.lock").open("a+") as lease, \
                 patch.object(launch, "gpu_idle"), \
                 patch.object(launch.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(launch, "wait_worker", side_effect=TimeoutError("limit")), \
                 patch.object(launch.os, "killpg") as kill, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.supervise(self.args("supervise", root, lease)), 2)
            spawn.assert_called_once()
            kill.assert_called_once_with(34567, signal.SIGTERM)
            result = launch.read_json(root / "summary.json")
            self.assertEqual(result["phase"], "failed")
            self.assertFalse(result["diagnostic_complete"])

    def test_waiter_respects_deadline_and_heartbeat(self):
        child = MagicMock(pid=444)
        child.wait.side_effect = subprocess.TimeoutExpired("worker", 30)
        with tempfile.TemporaryDirectory() as base, \
             patch.object(launch.time, "monotonic", side_effect=[0, 0, 30, 30, 45, 45]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                launch.wait_worker(child, 45, {}, base)
        self.assertEqual([call.kwargs["timeout"] for call in child.wait.call_args_list], [30, 15])

    def test_only_explicit_diagnostic_success_accepted_experiment_stays_incomplete(self):
        for worker_phase, diagnostic_complete, expected in [
                ("plain_window_complete", True, 0), ("running", False, 2)]:
            with self.subTest(phase=worker_phase), tempfile.TemporaryDirectory() as base:
                _, root = self.make_run(base)
                launch.write_json(root / "summary.json", dict(phase=worker_phase,
                    diagnostic_complete=diagnostic_complete, complete=False, test_accessed=False))
                child = MagicMock(pid=45678)
                child.wait.return_value = 0
                child.poll.return_value = 0
                with (Path(base) / ".severity_night.lock").open("a+") as lease, \
                     patch.object(launch, "gpu_idle"), \
                     patch.object(launch.subprocess, "Popen", return_value=child), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launch.supervise(self.args("supervise", root, lease)), expected)
                status = launch.read_json(root / "supervisor.json")
                self.assertFalse(status["complete"])
                self.assertFalse(status["experiment_complete"])

    def test_changed_runner_refused_before_worker_spawn(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "runner.pyz").write_bytes(b"changed")
            with patch.object(launch.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "changed after start"):
                    launch.validate_config(root)
                spawn.assert_not_called()

    def test_logs_tail_is_bounded_to_70_lines(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "diagnostic.log").write_text("".join(f"row {i}\n" for i in range(100)))
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                launch.main(["logs", "--output", str(root)])
            rows = captured.getvalue().splitlines()
            self.assertEqual(len(rows), 71)
            self.assertEqual(rows[1], "row 30")
            self.assertEqual(rows[-1], "row 99")


if __name__ == "__main__":
    unittest.main()
