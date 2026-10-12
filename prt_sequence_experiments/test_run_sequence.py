"""CPU tests for bounded branching and evidence gates; no BRIGHT/GPU access."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("run_sequence", Path(__file__).with_name("run_sequence.py"))
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def passed_report(evidence):
    return dict(format="SRAC_REGION_HEAD_ADAPT_EVAL_V1", complete=True,
                full_validation_evidence=True, image_count=450, expected_image_count=450, initial=True,
                smoke=False, training=False, test_accessed=False,
                initial_checks_passed=True, initial_parameters={"passed": True},
                initial_probability_checks={arm: dict(passed=True, checked_images=450, failed_images=0)
                                            for arm in ("p", "r", "t")},
                reproduction={"passed": True}, e0_metric_comparison={"passed": True},
                native_a1_replay_ok=True, immutable={"source": True},
                pool_guard={"restored": True, "failures": []},
                protocol_sha256=evidence["protocol_sha256"], source_sha256=evidence["original_source_sha256"])


def check_summary(evidence, mode="full", passed=True, failure=None):
    return dict(format=sequence.CHECK_FORMAT, mode=mode,
                complete=passed, full_order_passed=bool(passed and mode == "full"),
                diagnostic_only=True, full_validation_evidence=False,
                training=False, test_accessed=False, failure_kind=failure,
                scene_captured=mode == "capture", protocol_sha256=evidence["protocol_sha256"],
                original_source_sha256=evidence["original_source_sha256"])


def fake_failure(root):
    root = Path(root)
    names = ["image_%03d" % index for index in range(450)]
    protocol = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
                    full_validation_evidence=True, image_count=450, scope="all",
                    profile="legacy", pool_mode="native", smoke=False,
                    reproduction={"passed": True}, selected_images=names,
                    input_files_unchanged=True, code_unchanged=True,
                    frozen_evidence_unchanged=True, numerical_unchanged=True,
                    arguments={"joint_run": "/joint"})
    write_json(root / "fresh_e0/protocol.json", protocol)
    evidence = dict(protocol_sha256=sequence.digest(root / "fresh_e0/protocol.json"),
                    original_source_sha256="a" * 64)
    common = dict(format="SRAC_REGION_HEAD_ADAPT_V1", smoke=False, test_accessed=False,
                  protocol_sha256=evidence["protocol_sha256"], source_sha256=evidence["original_source_sha256"])
    write_json(root / "prt_fit/manifest.json", common)
    write_json(root / "prt_fit/summary.json", dict(common, complete=False, phase="initial_Val",
                                                   updates_per_arm={arm: 0 for arm in ("p", "r", "t")}))
    report = passed_report(evidence)
    report.update(complete=False, full_validation_evidence=False, initial_checks_passed=False)
    report["initial_probability_checks"]["p"].update(passed=False, failed_images=1)
    write_json(root / "prt_fit/initial_val/result.json", report)
    (root / "prt_fit/initial_val/images.jsonl").write_text(
        "".join(json.dumps({"image": name, "index": index}) + "\n" for index, name in enumerate(names)))
    return root


class SequenceContracts(unittest.TestCase):
    def prepare(self, td, continue_training=False):
        root = Path(td)
        (root / "logs").mkdir()
        args = sequence.parser().parse_args(["--continue-to-training"] if continue_training else [])
        evidence = dict(root="/failed", protocol="/fixed/protocol.json", protocol_sha256="p" * 64,
                        original_source_sha256="s" * 64)
        state = dict(complete=False, stages=[], checks=[], training_entry_started=False,
                     diagnostic_processes_run=0, training_blocked=True)
        return root, args, evidence, state

    def runner(self, evidence, outcomes, calls, reports_mutator=None):
        remaining = iter(outcomes)
        def run(command, log, _repo, _env):
            calls.append(command)
            Path(log).write_text("synthetic process output")
            if "--mode" not in command:
                return next(remaining)[0]
            mode = command[command.index("--mode") + 1]
            output = Path(command[command.index("--output") + 1])
            rc, failure = next(remaining)
            summary = check_summary(evidence, mode, rc == 0, failure)
            write_json(output / "check.json", summary)
            if mode == "full" and rc == 0:
                report = passed_report(evidence)
                if reports_mutator:
                    reports_mutator(report)
                write_json(output / "initial_val/result.json", report)
            return rc
        return run

    def test_two_passed_checks_default_no_training_no_root_cause_claim(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td)
            calls, waits = [], []
            rc = sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                         runner=self.runner(evidence, [(0, None), (0, None)], calls),
                                         gpu_wait=lambda: waits.append("wait"))
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(waits, ["wait", "wait"])
            self.assertTrue(state["eligible_for_training"])
            self.assertEqual(state["history_status"], "history_not_reproduced")
            self.assertFalse(state["root_cause_resolved"])
            self.assertFalse(state["training_entry_started"])

    def test_first_initial_gate_failure_fixed_capture_blocks_even_success(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            rc = sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                         runner=self.runner(evidence, [(1, "initial_probability_gate"), (0, None)], calls))
            self.assertEqual(rc, 1)
            modes = [command[command.index("--mode") + 1] for command in calls]
            self.assertEqual(modes, ["full", "capture"])
            self.assertTrue(state["training_blocked"])
            self.assertFalse(state["eligible_for_training"])
            self.assertFalse(state["training_entry_started"])
            self.assertEqual(state["diagnostic_processes_run"], 2)

    def test_second_full_gate_failure_no_third_capture_or_training(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            rc = sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                         runner=self.runner(evidence, [(0, None), (1, "initial_probability_gate")], calls))
            self.assertEqual(rc, 1)
            self.assertEqual(len(calls), 2)
            self.assertTrue(state["training_blocked"])
            self.assertFalse(state["training_entry_started"])

    def test_cuda_or_other_error_stops_no_capture(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            with self.assertRaisesRegex(RuntimeError, "outside"):
                sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                        runner=self.runner(evidence, [(1, "other")], calls))
            self.assertEqual(len(calls), 1)
            self.assertFalse(state["training_entry_started"])

    def test_opt_in_one_original_night_with_explicit_old_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            rc = sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                         runner=self.runner(evidence, [(0, None), (0, None), (0, None)], calls))
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 3)
            night_command = calls[-1]
            self.assertEqual(night_command[night_command.index("--protocol") + 1], evidence["protocol"])
            self.assertNotIn("--smoke", night_command)
            self.assertTrue(state["training_entry_started"])
            self.assertTrue(state["complete"])

    def test_night_failure_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            with self.assertRaisesRegex(RuntimeError, "no retry"):
                sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                        runner=self.runner(evidence, [(0, None), (0, None), (1, None)], calls))
            self.assertEqual(len(calls), 3)
            self.assertFalse(state["complete"])

    def test_zero_returncode_cannot_bypass_failed_formal_report(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td, True)
            calls = []
            with self.assertRaisesRegex(RuntimeError, "outside"):
                sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None, {},
                                        runner=self.runner(evidence, [(0, None)], calls,
                                                           lambda report: report.update(image_count=449)))
            self.assertEqual(len(calls), 1)
            self.assertTrue(state["training_blocked"])

    def test_strict_admission_requires_e0_all_immutable_and_all_arms(self):
        with tempfile.TemporaryDirectory() as td:
            root, _args, evidence, _state = self.prepare(td)
            summary = check_summary(evidence)
            path = root / "initial_val/result.json"
            for mutate in (
                lambda report: report["e0_metric_comparison"].update(passed=False),
                lambda report: report["immutable"].update(source=False),
                lambda report: report["initial_probability_checks"]["r"].update(failed_images=1),
                lambda report: report["initial_probability_checks"].pop("t"),
                lambda report: report.update(protocol_sha256="another"),
            ):
                report = passed_report(evidence)
                mutate(report)
                write_json(path, report)
                self.assertFalse(sequence.full_check_admitted(summary, root, evidence))

    def test_evidence_and_immutable_changes_block_before_new_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, state = self.prepare(td)
            source = root / "immutable.py"
            source.write_text("original")
            ledger = sequence.file_ledger([source])
            source.write_text("edited")
            runner = mock.Mock()
            with self.assertRaisesRegex(ValueError, "changed"):
                sequence.execute_checks(args, root, root, {}, state, evidence, "/joint", None,
                                        ledger, runner=runner)
            runner.assert_not_called()

    def test_archive_requires_only_initial_failure_and_exact_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = fake_failure(Path(td) / "failed")
            evidence = sequence.checked_failed_run(root)
            self.assertEqual(evidence["protocol"], str(root / "fresh_e0/protocol.json"))
            summary = sequence.read_json(root / "prt_fit/summary.json")
            summary["updates_per_arm"]["p"] = 1
            write_json(root / "prt_fit/summary.json", summary)
            with self.assertRaisesRegex(ValueError, "zero updates"):
                sequence.checked_failed_run(root)

    def test_archive_order_change_cannot_relabel_as_same_e0(self):
        with tempfile.TemporaryDirectory() as td:
            root = fake_failure(Path(td) / "failed")
            path = root / "prt_fit/initial_val/images.jsonl"
            rows = path.read_text().splitlines()
            rows.reverse()
            path.write_text("\n".join(rows))
            with self.assertRaisesRegex(ValueError, "order"):
                sequence.checked_failed_run(root)

    def test_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as td:
            with sequence.exclusive_lock(td):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    with sequence.exclusive_lock(td):
                        self.fail("duplicate lock acquired")
            with sequence.exclusive_lock(td):
                pass

    def test_default_checks_only_and_opt_in_are_mutually_exclusive(self):
        self.assertFalse(sequence.parser().parse_args([]).continue_to_training)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sequence.parser().parse_args(["--checks-only", "--continue-to-training"])

    def test_no_gpu_initialized_in_launcher_import(self):
        # This launcher intentionally exposes only stdlib orchestration.
        self.assertNotIn("torch", sequence.__dict__)

    def test_real_helper_cli_accepts_both_planned_modes(self):
        helper_spec = importlib.util.spec_from_file_location("sequence_cli_contract", Path(__file__).with_name("initial_sequence.py"))
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as td:
            root, args, evidence, _state = self.prepare(td)
            for mode in ("full", "capture"):
                command = sequence.check_command(args, evidence, root / "check", mode)
                parsed = helper.parser().parse_args(command[3:])
                self.assertEqual(parsed.mode, mode)
                self.assertEqual(parsed.protocol, evidence["protocol"])

    def test_real_helper_and_launcher_agree_on_failed_artifact_binding(self):
        helper_spec = importlib.util.spec_from_file_location("sequence_binding_contract", Path(__file__).with_name("initial_sequence.py"))
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as td:
            root = fake_failure(Path(td) / "failed")
            evidence = sequence.checked_failed_run(root)
            _paths, hashes, source, names = helper.bind_failed_evidence(
                root, evidence["protocol"], Path(td) / "new", Path(td) / "repo")
            self.assertEqual(hashes["protocol"], evidence["protocol_sha256"])
            self.assertEqual(source, evidence["original_source_sha256"])
            self.assertEqual(len(names), 450)

    def test_plan_only_creates_no_outputs_or_processes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            failed = fake_failure(root / "failed")
            repo = root / "repo"
            (repo / "bda_benchmark").mkdir(parents=True)
            parent = root / "outputs"
            source = dict(joint_run="/joint", cache=str(root / "cache"),
                          dataset=str(root / "dataset"), device="cuda:0")
            night = SimpleNamespace(source_arguments=mock.Mock(return_value=source), run_logged=mock.Mock())
            with mock.patch.object(sequence, "load_night_tools", return_value=(night, {})), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                rc = sequence.main(["--repo", str(repo), "--python", sys.executable,
                                    "--failed-run", str(failed), "--output-parent", str(parent), "--plan-only"])
            self.assertEqual(rc, 0)
            self.assertFalse(parent.exists())
            night.run_logged.assert_not_called()
            self.assertIn("check_02_if_first_initial_gate_failed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
