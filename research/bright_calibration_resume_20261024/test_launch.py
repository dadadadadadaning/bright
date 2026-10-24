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
import zipfile
from unittest.mock import MagicMock, patch

import launch


class LauncherContracts(unittest.TestCase):
    def test_source_snapshot_contains_all_modules_and_imports_help(self):
        with tempfile.TemporaryDirectory() as base:
            bundle = Path(base) / 'runner.pyz'
            with patch.object(launch.sys, 'argv', [str(Path(launch.__file__))]):
                launch.snapshot_runner(bundle)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()),
                    {'__main__.py','launch.py','helpers.py','preflight.py','resume.py','lean_adapter.py'})
            result = subprocess.run([launch.sys.executable, '-B', str(bundle), '--help'],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def make_run(self, base):
        base = Path(base)
        parent = base / "night_3407" / "train_calibration"
        parent.mkdir(parents=True)
        (parent / "checkpoints").mkdir()
        (parent / "checkpoints/checkpoint_000800.pt").write_bytes(b"checkpoint")
        root = base / "calibration_resume_case"
        root.mkdir()
        (root / "runner.pyz").write_bytes(b"diagnostic snapshot")
        launch.write_json(root / "config.json", dict(format=launch.FORMAT,
            parent=str(parent), package=str(base / "package"), output=str(root),
            timeout_seconds=60, runner_sha256=launch.digest(root / "runner.pyz")))
        (root / "config.json.sha256").write_text(launch.digest(root / "config.json") + "  config.json\n")
        launch.write_json(root / "summary.json", dict(phase="diagnostic_starting",
            complete=False, recovery_complete=False, test_accessed=False))
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
        self.assertEqual(launch.study_root("/tmp/study/night/train_calibration"), Path("/tmp/study"))

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
            other = Path(base) / "calibration_resume_newer"
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
                with patch.object(launch.resume, "run_worker") as run:
                    self.assertEqual(launch.worker(args), 0)
                run.assert_called_once_with(args)
                self.assertEqual(args.parent, parent)
                self.assertEqual(args.output, root)
                self.assertEqual(args.package, Path(base) / "package")
                self.assertFalse(hasattr(args, "baseline"))
                self.assertFalse(hasattr(args, "plain"))

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

            captured = io.StringIO()
            with patch.object(launch.preflight, "check_inputs") as preflight, patch.object(launch, "gpu_idle"), \
                 patch.object(launch, "snapshot_runner", side_effect=snapshot), \
                 patch.object(launch.subprocess, "Popen", side_effect=spawn_checked) as spawn, \
                 contextlib.redirect_stdout(captured):
                self.assertEqual(launch.start(args), 0)
            preflight.assert_called_once_with(Path(base), parent)
            output = Path(launch.read_json(Path(base) / launch.MARKER)["output"])
            self.assertNotEqual(output, original)
            self.assertTrue(output.name.startswith("calibration_resume_"))
            self.assertEqual((original / "evidence.txt").read_text(), "preserve")
            self.assertEqual(spawn.call_count, 1)
            options = spawn.call_args.kwargs
            self.assertTrue(options["start_new_session"])
            self.assertEqual(len(options["pass_fds"]), 1)
            self.assertEqual(options["env"]["CUDA_LAUNCH_BLOCKING"], "1")
            config = launch.validate_config(output)
            self.assertEqual(config["variant"], "calibration")
            self.assertEqual(config["execution_change"], launch.EXECUTION_CHANGE)
            self.assertEqual(config["restored_step"], 800)
            self.assertEqual(config["new_updates"], 1700)
            self.assertEqual(config["new_val_steps"], [1000,1500,2000,2500])
            self.assertEqual(config["disk_headroom"]["expected_new_checkpoints"], 34)
            self.assertFalse(config["automatic_retry"])
            self.assertFalse(config["test_accessed"])
            self.assertNotIn("baseline", config)
            self.assertNotIn("plain", config)
            for text in ("800→2500", "新增1700", "1000/1500/2000/2500", "Train450", "不运行Test"):
                self.assertIn(text, captured.getvalue())

    def test_start_rejects_over_budget_or_reused_output(self):
        for flags, message in [(["--timeout-seconds", "86401"], "1..86400"),
                               (["--output", "/tmp/old"], "new calibration_resume")]:
            with self.subTest(flags=flags), patch.object(launch.preflight, "check_inputs") as read:
                with self.assertRaisesRegex(ValueError, message):
                    launch.start(launch.parser().parse_args(["start", *flags]))
                read.assert_not_called()

    def test_failed_worker_stops_without_retry_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "first_failure.json").write_text('{"original":"assert"}')
            launch.write_json(root / "summary.json", dict(phase="failed", error="first CUDA assert",
                complete=False, recovery_complete=False, test_accessed=False))
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
            self.assertFalse(result["recovery_complete"])
            self.assertTrue(result["supervisor_interrupted_worker"])
            self.assertTrue(result["failed_step_optimizer_execution_unknown"])

    def test_late_supervisor_failure_preserves_completed_evaluation_execution(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            launch.write_json(root / "summary.json", dict(phase="final_integrity", complete=False,
                training_complete=True, validation_complete=True, fixed_train_complete=True,
                full_validation_evidence=False, failed_step_optimizer_execution_unknown=False))
            launch.failed_summary(root, "final integrity failed")
            result = launch.read_json(root / "summary.json")
            self.assertTrue(result["training_complete"])
            self.assertTrue(result["validation_complete"])
            self.assertTrue(result["fixed_train_complete"])
            self.assertFalse(result["full_validation_evidence"])
            self.assertFalse(result["complete"])
            self.assertFalse(result["failed_step_optimizer_execution_unknown"])

    def test_waiter_respects_deadline_and_heartbeat(self):
        child = MagicMock(pid=444)
        child.wait.side_effect = subprocess.TimeoutExpired("worker", 30)
        with tempfile.TemporaryDirectory() as base, \
             patch.object(launch.time, "monotonic", side_effect=[0, 0, 30, 30, 45, 45]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                launch.wait_worker(child, 45, {}, base)
        self.assertEqual([call.kwargs["timeout"] for call in child.wait.call_args_list], [30, 15])

    def completed_summary(self):
        evaluations = {
            f"val_step_{step:06d}": dict(step=step, split="val", image_count=450,
                complete=True, final_integrity_pending=False, execution_change=launch.EXECUTION_CHANGE)
            for step in launch.EVALUATION_STEPS}
        evaluations["fixed_train_endpoint"] = dict(step=2500, split="fixed_train", image_count=450,
            complete=True, final_integrity_pending=False, execution_change=launch.EXECUTION_CHANGE)
        return dict(phase="complete", complete=True, recovery_complete=True,
            variant="calibration", execution_change=launch.EXECUTION_CHANGE,
            training_complete=True, validation_complete=True, fixed_train_complete=True,
            full_validation_evidence=True, full_test_evidence=False, test_accessed=False,
            last_complete_step=2500, new_complete_updates=1700, historical_failure_resolved=False,
            inherited_evaluations_reference_only=True, study_complete=False, evaluations=evaluations,
            error=None)

    def test_only_complete_calibration_training_and_new_validation_accepted(self):
        for good in (True, False):
            with self.subTest(good=good), tempfile.TemporaryDirectory() as base:
                _, root = self.make_run(base)
                summary = self.completed_summary()
                summary.update(validation_complete=good, full_validation_evidence=good)
                launch.write_json(root / "summary.json", summary)
                child = MagicMock(pid=45678)
                child.wait.return_value = 0
                child.poll.return_value = 0
                previous_hup = signal.getsignal(signal.SIGHUP)

                def spawn_checked(*args, **kwargs):
                    self.assertTrue(fcntl.fcntl(kwargs['stdout'].fileno(), fcntl.F_GETFL) & os.O_APPEND)
                    self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
                    self.assertEqual(kwargs["env"]["CUDA_LAUNCH_BLOCKING"], "1")
                    self.assertTrue(kwargs["start_new_session"])
                    self.assertEqual(len(kwargs["pass_fds"]), 1)
                    return child

                with (Path(base) / ".severity_night.lock").open("a+") as lease, \
                     patch.object(launch, "gpu_idle"), \
                     patch.object(launch.subprocess, "Popen", side_effect=spawn_checked), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launch.supervise(self.args("supervise", root, lease)), 0 if good else 2)
                self.assertEqual(signal.getsignal(signal.SIGHUP), previous_hup)
                status = launch.read_json(root / "supervisor.json")
                self.assertEqual(status["complete"], good)
                self.assertFalse(status["full_test_evidence"])
                self.assertFalse(status["original_experiment_complete"])
                self.assertFalse(status["experiment_complete"])
                self.assertFalse(status["study_complete"])
                self.assertTrue(status["inherited_evaluations_reference_only"])
                if not good:
                    self.assertTrue(launch.read_json(root / "summary.json")["training_complete"])

    def test_disk_check_blocks_before_new_worker(self):
        with tempfile.TemporaryDirectory() as base:
            parent, root = self.make_run(base)
            with patch.object(launch.shutil, 'disk_usage', return_value=MagicMock(free=0)):
                with self.assertRaisesRegex(RuntimeError, 'disk headroom'):
                    launch.disk_headroom(root, parent)

    def test_changed_runner_refused_before_worker_spawn(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "runner.pyz").write_bytes(b"changed")
            with patch.object(launch.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "changed after start"):
                    launch.validate_config(root)
                spawn.assert_not_called()

    def test_changed_configuration_refused_before_worker_spawn(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            config = launch.read_json(root / "config.json")
            config["parent"] = str(Path(base) / "another_parent")
            launch.write_json(root / "config.json", config)
            with patch.object(launch.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "configuration changed after start"):
                    launch.validate_config(root)
                spawn.assert_not_called()

    def test_completion_rejects_wrong_variant_execution_count_or_evaluations(self):
        for field, value in (("variant", "original"), ("execution_change", "unchanged"),
                             ("new_complete_updates", 2350), ("fixed_train_complete", False),
                             ("test_accessed", True), ("full_test_evidence", True),
                             ("inherited_evaluations_reference_only", False), ("study_complete", True)):
            with self.subTest(field=field):
                summary = self.completed_summary()
                summary[field] = value
                with self.assertRaisesRegex(ValueError, "completion evidence differs"):
                    launch.validate_success(summary)
        summary = self.completed_summary()
        del summary["evaluations"]["val_step_001000"]
        with self.assertRaisesRegex(ValueError, "exactly four new"):
            launch.validate_success(summary)
        summary = self.completed_summary()
        summary["evaluations"]["val_step_000500"] = dict(summary["evaluations"]["val_step_001000"])
        with self.assertRaisesRegex(ValueError, "exactly four new"):
            launch.validate_success(summary)
        for field, value in (("complete", False), ("final_integrity_pending", True),
                             ("image_count", 449), ("step", 500), ("split", "test"),
                             ("execution_change", "unchanged")):
            with self.subTest(evaluation_field=field):
                summary = self.completed_summary()
                summary["evaluations"]["val_step_001000"][field] = value
                with self.assertRaisesRegex(ValueError, "evaluation evidence differs"):
                    launch.validate_success(summary)

    def test_check_uses_only_package_and_parent_without_gpu_access(self):
        args = launch.parser().parse_args(["check", "--package", "/tmp/package", "--parent", "/tmp/parent"])
        with patch.object(launch.preflight, "check_inputs") as preflight, \
             patch.object(launch, "gpu_idle") as gpu, contextlib.redirect_stdout(io.StringIO()):
            launch.check(args)
        preflight.assert_called_once_with(Path("/tmp/package"), Path("/tmp/parent"))
        gpu.assert_not_called()

    def test_default_budget_and_source_parent(self):
        args = launch.parser().parse_args(["start"])
        self.assertEqual(args.timeout_seconds, 43200)
        self.assertEqual(launch.MAX_SECONDS, 86400)
        self.assertEqual(launch.HEARTBEAT_SECONDS, 30)
        self.assertEqual(args.parent, Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1/night_3407_z_yxvq4s/train_calibration"))
        for option in ("--observed", "--baseline", "--plain"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.parser().parse_args(["start", option, "/tmp/unused"])

    def test_logs_tail_is_bounded_to_70_lines(self):
        with tempfile.TemporaryDirectory() as base:
            _, root = self.make_run(base)
            (root / "train.log").write_text("".join(f"row {i}\n" for i in range(100)))
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                launch.main(["logs", "--output", str(root)])
            rows = captured.getvalue().splitlines()
            self.assertEqual(len(rows), 71)
            self.assertEqual(rows[1], "row 30")
            self.assertEqual(rows[-1], "row 99")


if __name__ == "__main__":
    unittest.main()
