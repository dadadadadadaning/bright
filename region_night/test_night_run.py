"""CPU launcher contracts; no BRIGHT data, GPU, model fitting or Test access."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("night_run", Path(__file__).with_name("night_run.py"))
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)


def fake_joint(root):
    run = root / "joint"
    run.mkdir()
    args = dict(steps=5000, seed=3407, device="cuda:0", workers=4)
    for key in ("a1_checkpoint", "checkpoint", "cache", "metric_checkpoint", "dataset", "train_list", "val_list"):
        path = root / key
        if key in ("cache", "dataset"):
            path.mkdir()
        else:
            path.write_text("synthetic fixture")
        args[key] = str(path)
    manifest = dict(format="SRAC_SNAPSHOT_JOINT123_V1", arguments=args, smoke=False, probe=False, test_accessed=False)
    summary = dict(complete=True, run_complete=True, frozen_fingerprints_unchanged=True,
                   smoke=False, probe=False, test_accessed=False, train_image_counts={a: 5000 for a in ("j1", "j2", "j3")})
    for name, value in (("manifest.json", manifest), ("summary.json", summary)):
        (run / name).write_text(json.dumps(value))
    for name in ("j1_last.pth", "j2_last.pth", "j3_last.pth", "validation.jsonl"):
        (run / name).write_text("synthetic fixture")
    return run


class NightContracts(unittest.TestCase):
    def test_joint_requires_completed_equal_budget_not_probe(self):
        with tempfile.TemporaryDirectory() as td:
            joint = fake_joint(Path(td))
            self.assertEqual(night.source_arguments(joint)["seed"], 3407)
            summary = night.read_json(joint / "summary.json")
            summary["train_image_counts"]["j3"] = 4999
            (joint / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "5000"):
                night.source_arguments(joint)

    def test_probe_and_missing_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            joint = fake_joint(Path(td))
            manifest = night.read_json(joint / "manifest.json")
            manifest["probe"] = True
            (joint / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "probe"):
                night.source_arguments(joint)
            manifest["probe"] = False
            (joint / "manifest.json").write_text(json.dumps(manifest))
            (joint / "j3_last.pth").unlink()
            with self.assertRaises(FileNotFoundError):
                night.source_arguments(joint)

    def test_original_relative_artifact_paths_use_repository_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            joint = fake_joint(root)
            manifest = night.read_json(joint / "manifest.json")
            for key in ("a1_checkpoint", "checkpoint", "cache", "metric_checkpoint", "dataset", "train_list", "val_list"):
                manifest["arguments"][key] = key
            (joint / "manifest.json").write_text(json.dumps(manifest))
            result = night.source_arguments(joint, root)
            self.assertEqual(result["a1_checkpoint"], str(root / "a1_checkpoint"))

    def test_fresh_e0_order_and_original_argument_binding(self):
        args = night.parser().parse_args(["--repo", "/repo path", "--prt-only"])
        source = dict(joint_run="/joint path", a1_checkpoint="/a1", checkpoint="/front", cache="/cache",
                      metric_checkpoint="/metric", dataset="/dataset", train_list="/train", val_list="/val",
                      device="cuda:0", workers=4, seed=82)
        plan, protocol = night.build_plan(args, Path("/output"), source)
        names = [name for name, _ in plan]
        self.assertEqual(names, ["source_check", "prt_preflight", "fresh_e0", "prt_train", "prt_evaluate", "prt_report"])
        cmd = dict(plan)["fresh_e0"]
        self.assertEqual(cmd[cmd.index("--joint_run") + 1], "/joint path")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "82")
        self.assertNotIn("--smoke", cmd)
        for key, value in (("--scope", "all"), ("--profile", "legacy"), ("--pool_mode", "native")):
            self.assertEqual(cmd[cmd.index(key) + 1], value)
        self.assertEqual(protocol, "/output/fresh_e0/protocol.json")

    def test_explicit_protocol_not_replaced_with_fresh_or_latest(self):
        args = night.parser().parse_args(["--protocol", "/fixed/protocol.json", "--prt-only"])
        plan, protocol = night.build_plan(args, Path("/output"), {"joint_run": "/joint"})
        self.assertNotIn("fresh_e0", [name for name, _ in plan])
        self.assertEqual(protocol, "/fixed/protocol.json")
        self.assertEqual(dict(plan)["prt_train"][dict(plan)["prt_train"].index("--protocol") + 1], protocol)

    def test_failure_stops_dependent_training_and_records_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir()
            state = dict(complete=False, stages=[], protocol="not used")
            called = []
            def runner(cmd, *_args):
                called.append(cmd[0])
                return 2
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                night.execute_plan([("preflight", ["fail"]), ("train", ["never"])], root, root, {}, state, runner)
            self.assertEqual(called, ["fail"])
            recorded = night.read_json(root / "workflow_state.json")
            self.assertFalse(recorded["complete"])
            self.assertEqual(recorded["stages"][0]["returncode"], 2)

    def test_serial_success_waits_once_before_gpu_and_records_fresh_digest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "logs").mkdir()
            protocol = root / "protocol.json"
            state = dict(complete=False, stages=[], protocol=str(protocol))
            called = []
            def runner(cmd, *_args):
                called.append(cmd[0])
                if cmd[0] == "fresh_e0":
                    protocol.write_text("new source evidence")
                return 0
            plan = [(name, [name]) for name in ("preflight", "fresh_e0", "prt_train", "prt_evaluate", "residual_train")]
            wait = lambda: called.append("waiting_gpu")
            night.execute_plan(plan, root, root, {}, state, runner, gpu_wait=wait)
            self.assertEqual(called, ["preflight", "waiting_gpu", "fresh_e0", "prt_train", "prt_evaluate", "residual_train"])
            recorded = night.read_json(root / "workflow_state.json")
            self.assertTrue(recorded["complete"])
            self.assertEqual(recorded["protocol_sha256"], night.digest(protocol))
            self.assertTrue(all(row["status"] == "passed" for row in recorded["stages"]))

    def test_lock_rejects_second_launcher(self):
        with tempfile.TemporaryDirectory() as td:
            with night.exclusive_lock(td):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    with night.exclusive_lock(td):
                        self.fail("second lock acquired")
            with night.exclusive_lock(td):
                pass

    def test_plan_only_has_no_output_or_gpu_access(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            joint = fake_joint(root)
            repo = root / "repo"
            repo.mkdir()
            output = root / "new_output"
            with contextlib.redirect_stdout(io.StringIO()) as stdout, mock.patch.object(night, "run_logged") as run:
                rc = night.main(["--repo", str(repo), "--python", sys.executable, "--joint-run", str(joint),
                                 "--output-parent", str(output), "--prt-only", "--plan-only"])
            self.assertEqual(rc, 0)
            self.assertFalse(output.exists())
            run.assert_not_called()
            self.assertIn("fresh_e0_not_original_protocol_restoration", stdout.getvalue())

    def test_wait_busy_process_then_idle_never_signals_other_pid(self):
        outputs = iter(["1234\n", ""])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(night.os, "killpg") as kill:
            pauses = []
            night.wait_gpu_idle("cuda:0", query=lambda: next(outputs), pause=pauses.append)
        self.assertEqual(pauses, [30])
        kill.assert_not_called()

    def test_gpu_visibility_and_invalid_query_are_fail_closed(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,GPU-abc"}, clear=True):
            self.assertEqual(night.gpu_selector("cuda:1"), "GPU-abc")
            with self.assertRaises(ValueError):
                night.gpu_selector("cuda:2")
            with self.assertRaisesRegex(ValueError, "cannot assume idle"):
                night.wait_gpu_idle("cuda:0", query=lambda: "permission denied")

    def test_wait_timeout_does_not_train_or_kill(self):
        times = iter([0, 61])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(night.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                night.wait_gpu_idle("cuda:0", timeout=60, query=lambda: "9", now=lambda: next(times))
        kill.assert_not_called()

    def test_logged_command_has_exit_code_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "step.log"
            rc = night.run_logged([sys.executable, "-c", "print('fixture failure'); raise SystemExit(7)"], log, td, os.environ)
            self.assertEqual(rc, 7)
            self.assertIn("fixture failure", log.read_text())

    def test_child_interruption_only_stops_owned_process_group(self):
        child = mock.Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("own child", 10), 0]
        with mock.patch.object(night.os, "killpg") as kill:
            night.stop_child(child)
        self.assertEqual(kill.call_args_list, [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)])


if __name__ == "__main__":
    unittest.main()
