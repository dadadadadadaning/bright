"""CPU/stdlib tests; no BRIGHT data, CUDA use or training subprocesses."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import launch_joint123_when_ready as launcher


class LauncherContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.parent = self.root / "parent"
        self.parent.mkdir()

    def write_json(self, relative, value):
        path = self.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def complete(self):
        self.write_json("front/summary.json", dict(complete=True, smoke=False, test_accessed=False))
        self.write_json("train_memory/manifest.json", dict(
            complete=True, smoke=False, test_accessed=False, split="train", image_count=2918,
            shards=[{}] * 2918))
        for relative in ("front/best.pth", "train_memory/bank.pt"):
            (self.parent / relative).write_bytes(b"placeholder; never loaded")

    def test_missing_artifacts_wait_only_while_parent_alive(self):
        ready, reason = launcher.artifacts_ready(self.parent)
        self.assertFalse(ready)
        self.assertFalse(launcher.decision(ready, reason, True, [])[0])
        with self.assertRaisesRegex(ValueError, "exited.*incomplete"):
            launcher.decision(ready, reason, False, [])

    def test_best_checkpoint_exists_does_not_imply_completion(self):
        (self.parent / "front").mkdir()
        (self.parent / "front/best.pth").write_bytes(b"step=2000")
        self.assertFalse(launcher.artifacts_ready(self.parent)[0])

    def test_in_progress_cache_and_partial_json_wait(self):
        self.complete()
        path = self.parent / "train_memory/manifest.json"
        data = json.loads(path.read_text())
        data["complete"] = False
        path.write_text(json.dumps(data))
        self.assertFalse(launcher.artifacts_ready(self.parent)[0])
        path.write_text('{"complete":')
        self.assertFalse(launcher.artifacts_ready(self.parent)[0])

    def test_completed_parent_still_waits_for_pipeline_and_gpu(self):
        self.complete()
        ready, reason = launcher.artifacts_ready(self.parent)
        self.assertTrue(ready)
        self.assertFalse(launcher.decision(ready, reason, True, [])[0])
        self.assertFalse(launcher.decision(ready, reason, False, [123])[0])
        self.assertTrue(launcher.decision(ready, reason, False, [])[0])

    def test_complete_markers_require_both_weights_and_bank(self):
        self.complete()
        (self.parent / "train_memory/bank.pt").write_bytes(b"")
        self.assertFalse(launcher.artifacts_ready(self.parent)[0])

    def test_smoke_test_access_and_wrong_split_are_rejected(self):
        self.complete()
        path = self.parent / "train_memory/manifest.json"
        original = json.loads(path.read_text())
        for key, value in (("smoke", True), ("test_accessed", True), ("split", "val"),
                           ("image_count", 2), ("shards", [{}])):
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(original, **{key: value})))
                with self.assertRaises(ValueError):
                    launcher.artifacts_ready(self.parent)

    def process(self, pid=123, state="S", ticks=900, run_root=None, worker=False):
        proc = self.root / "proc" / str(pid)
        proc.mkdir(parents=True, exist_ok=True)
        # stat fields 3..22; comm deliberately contains a closing parenthesis.
        fields = [state] + ["0"] * 18 + [str(ticks)]
        (proc / "stat").write_text(f"{pid} (bash (queue)) " + " ".join(fields))
        command = (b"python\0train_front\0" if worker else
                   b"bash\0bda_benchmark/experiments/damageformer_srac_transformer_v1/run.sh\0pipeline\0")
        (proc / "cmdline").write_bytes(command)
        (proc / "environ").write_bytes(b"RUN_ROOT=" + str(run_root or self.parent).encode() + b"\0")
        return self.root / "proc"

    def test_parent_pid_identity_detects_reuse_and_zombie(self):
        proc_root = self.process()
        first = launcher.verify_parent_process(123, self.parent.resolve(), proc_root)
        self.assertEqual(first, (123, "900"))
        self.process(ticks=901)
        self.assertNotEqual(first, launcher.process_identity(123, proc_root))
        self.process(state="Z")
        self.assertIsNone(launcher.process_identity(123, proc_root))
        self.assertIsNone(launcher.process_identity(999, proc_root))

    def test_worker_and_wrong_parent_run_rejected(self):
        proc_root = self.process(worker=True)
        with self.assertRaisesRegex(ValueError, "not a worker"):
            launcher.verify_parent_process(123, self.parent.resolve(), proc_root)
        self.process(run_root=self.root / "other_run")
        with self.assertRaisesRegex(ValueError, "does not match"):
            launcher.verify_parent_process(123, self.parent.resolve(), proc_root)

    def test_gpu_guard_is_fail_closed(self):
        with patch.object(launcher.subprocess, "run") as call:
            call.return_value.stdout = "123\n456\n123\n"
            self.assertEqual(launcher.gpu_processes(), [123, 456])
            call.return_value.stdout = ""
            self.assertEqual(launcher.gpu_processes(), [])
            call.return_value.stdout = "N/A"
            with self.assertRaises(ValueError):
                launcher.gpu_processes()
            call.side_effect = subprocess.CalledProcessError(1, ["nvidia-smi"])
            with self.assertRaises(subprocess.SubprocessError):
                launcher.gpu_processes()

    def test_output_is_new_and_outside_parent(self):
        a1 = self.root / "a1.pth"
        a1.write_bytes(b"fake")
        script = self.root / "bda_benchmark/experiments/damageformer_srac_joint123_v1/run.sh"
        script.parent.mkdir(parents=True)
        script.write_text("fake; never executed")
        output = self.root / "joint_family" / "new_run"
        self.assertEqual(launcher.validate_paths(self.root, self.parent, a1, output), script)
        for wrong in (self.parent, self.parent / "inside", self.root):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                launcher.validate_paths(self.root, self.parent, a1, wrong)
        output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            launcher.validate_paths(self.root, self.parent, a1, output)

    def mock_main(self):
        """Fake all Linux locks/process launches: safe on a Windows CPU host."""
        self.complete()
        a1 = self.root / "a1.pth"
        a1.write_bytes(b"fake; never loaded")
        script = self.root / "bda_benchmark/experiments/damageformer_srac_joint123_v1/run.sh"
        script.parent.mkdir(parents=True)
        script.write_text("fake; never executed")
        output = self.root / "joint_family" / "new_run"
        argv = ["launcher", "--project", str(self.root), "--parent-run", str(self.parent),
                "--a1-checkpoint", str(a1), "--output", str(output), "--parent-pid", "123"]
        patches = [
            patch.object(launcher.sys, "argv", argv),
            patch.object(launcher.sys, "platform", "linux"),
            patch.dict("sys.modules", fcntl=SimpleNamespace(flock=lambda *a: None, LOCK_EX=2, LOCK_NB=4)),
            patch.object(launcher, "verify_parent_process", return_value=None),
            patch.object(launcher, "log"),
            patch.object(launcher.time, "sleep"),
            patch.object(launcher.os, "set_inheritable"),
            patch.object(launcher.os, "chdir"),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        return output

    def test_ready_runs_strict_checks_before_pipeline_without_creating_output(self):
        output = self.mock_main()
        with patch.object(launcher, "gpu_processes", return_value=[]), \
                patch.object(launcher.subprocess, "run") as check, \
                patch.object(launcher.os, "execvpe") as execute:
            launcher.main()
            self.assertEqual([call.args[0][-1] for call in check.call_args_list], ["preflight", "check"])
            self.assertEqual(execute.call_args.args[1][-1], "pipeline")
            self.assertFalse(output.exists())
            launcher.os.set_inheritable.assert_called_once()

    def test_strict_check_failure_never_executes_pipeline(self):
        self.mock_main()
        with patch.object(launcher, "gpu_processes", return_value=[]), \
                patch.object(launcher.subprocess, "run",
                             side_effect=[None, subprocess.CalledProcessError(1, "check")]), \
                patch.object(launcher.os, "execvpe") as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main()
            execute.assert_not_called()

    def test_busy_gpu_after_checks_returns_to_waiting_then_launches_once(self):
        self.mock_main()
        with patch.object(launcher, "gpu_processes", side_effect=[[], [444], [], []]), \
                patch.object(launcher.subprocess, "run") as check, \
                patch.object(launcher.os, "execvpe") as execute:
            launcher.main()
            self.assertEqual(check.call_count, 4)  # Revalidate after GPU availability changes.
            execute.assert_called_once()
            launcher.time.sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
