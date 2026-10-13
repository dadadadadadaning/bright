"""Pipeline tests use saved synthetic evidence, never a model or CUDA."""
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

from bda_benchmark.experiments.damageformer_srac_memory_rebuild_v3 import run_pipeline as pipe


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.protocol = self.base / "source_e0.json"
        self.save(self.protocol, dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
            full_validation_evidence=True, image_count=450, smoke=False, training=False,
            test_accessed=False, arguments={"device": "cuda:0"}))
        self.root = self.base / "fresh_root"
        self.calls = []

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def cli(self, stage="smoke", **overrides):
        args = pipe.parser().parse_args(["--stage", stage, "--protocol", str(self.protocol), "--root", str(self.root)])
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def fake_build(self, folder, smoke):
        folder.mkdir()
        bank = folder / "paired_bank.pt"
        bank.write_bytes(b"synthetic-FP32-paired-keys-and-labels")
        protocol = dict(format=pipe.PROTOCOL_FORMAT, source_sha256="source-code-sha",
            protocol_sha256=pipe.digest(self.protocol), historical_replay_required=False,
            historical_fp32_reproduction_claim=False)
        self.save(folder / "protocol.json", protocol)
        count = 4 if smoke else 200
        summary = dict(format=pipe.BANK_FORMAT, complete=True, smoke=smoke, train_only=True,
            training=False, test_accessed=False, immutable={"all": True}, image_count=count,
            extraction_forward_count=count, paired_bank_sha256=pipe.digest(bank),
            rebuild_protocol_sha256=pipe.digest(folder / "protocol.json"))
        manifest = dict(summary, source_sha256="source-code-sha", protocol_sha256=pipe.digest(self.protocol),
            fresh_label_checks_passed=True, same_pass_q_checks_passed=True,
            saved_fp32_roundtrip_exact=True, pool_ok=True,
            t128_storage="torch.float32", f256_storage="torch.float32", label_storage="torch.float32")
        self.save(folder / "manifest.json", manifest)
        self.save(folder / "summary.json", summary)

    def fake_diagnose(self, folder, smoke):
        folder.mkdir()
        keys = self.root / ("keys_smoke" if smoke else "keys")
        summary = dict(format=pipe.DIAG_FORMAT, complete=not smoke, execution_complete=True,
            full_validation_evidence=not smoke, smoke=smoke, image_count=2 if smoke else 450,
            training=False, test_accessed=False, immutable={"all": True},
            validation={name: True for name in pipe.CHECKS})
        result = dict(summary, historical_reproduction_claim=False,
            labelvote_is_untrained_diagnostic=True, paired_bank_smoke=smoke,
            source_sha256="source-code-sha", source_anchor_protocol_sha256=pipe.digest(self.protocol),
            paired_bank_sha256=pipe.digest(keys / "paired_bank.pt"),
            paired_manifest_sha256=pipe.digest(keys / "manifest.json"),
            rebuild_protocol_sha256=pipe.digest(keys / "protocol.json"))
        self.save(folder / "summary.json", summary)
        self.save(folder / "result.json", result)
        (folder / "predictions").mkdir()
        (folder / "predictions" / "0000.npz").write_bytes(b"synthetic prediction evidence")

    def child(self, command, log):
        stage = log.stem
        self.calls.append(stage)
        log.write_text("Synthetic stage log: " + stage)
        if stage.startswith("keys"):
            self.fake_build(self.root / stage, stage.endswith("_smoke"))
        elif stage.startswith("diagnose"):
            self.fake_diagnose(self.root / stage, stage.endswith("_smoke"))
        return 0

    def execute(self, cli=None, child=None):
        with patch.object(pipe, "code_hash", return_value="source-code-sha"), \
                patch.object(pipe, "ensure_idle"), patch.object(pipe, "run_child", side_effect=child or self.child), \
                contextlib.redirect_stdout(io.StringIO()):
            pipe.run_pipeline(cli or self.cli())

    def test_smoke_success_records_exact_evidence_and_no_formal_claim(self):
        self.execute()
        state = pipe.read_json(self.root / "pipeline_state.json")
        self.assertEqual(self.calls, list(pipe.SMOKE_STAGES))
        self.assertIs(state["smoke_complete"], True)
        self.assertIs(state["complete"], False)
        self.assertIs(state["training"], False)
        self.assertIs(state["test_accessed"], False)
        self.assertIs(state["historical_replay_required"], False)
        with patch.object(pipe, "code_hash", return_value="source-code-sha"):
            pipe.verify_ledger(self.root, state)

    def test_full_new_root_runs_smoke_then_formal450(self):
        self.execute(self.cli("full"))
        self.assertEqual(self.calls, list(pipe.SMOKE_STAGES + pipe.FULL_STAGES))
        state = pipe.read_json(self.root / "pipeline_state.json")
        self.assertIs(state["complete"], True)
        self.assertIs(state["full_validation_evidence"], True)
        self.assertEqual(state["stages"]["diagnose"]["gate"]["image_count"], 450)

    def test_explicit_smoke_root_can_continue_once_to_full(self):
        self.execute()
        self.calls.clear()
        self.execute(self.cli("full"))
        self.assertEqual(self.calls, list(pipe.FULL_STAGES))
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "completed smoke provenance"):
            self.execute(self.cli("full"))
        self.assertEqual(self.calls, [])

    def test_zero_return_code_without_bank_gate_blocks_diagnosis(self):
        def bad(command, log):
            code = self.child(command, log)
            path = self.root / "keys_smoke" / "summary.json"
            summary = pipe.read_json(path)
            summary["complete"] = False
            self.save(path, summary)
            return code
        with self.assertRaisesRegex(ValueError, "completion gate"):
            self.execute(child=bad)
        self.assertEqual(self.calls, ["keys_smoke"])
        state = pipe.read_json(self.root / "pipeline_state.json")
        self.assertEqual(state["stages"]["keys_smoke"]["status"], "failed")
        self.assertIs(state["complete"], False)
        self.assertTrue((self.root / "pipeline_traceback.txt").is_file())

    def test_child_failure_blocks_every_later_stage_and_preserves_log(self):
        def failed(command, log):
            self.calls.append(log.stem)
            log.write_text("original failure detail")
            return 9
        with self.assertRaisesRegex(RuntimeError, "exited 9"):
            self.execute(child=failed)
        self.assertEqual(self.calls, ["keys_smoke"])
        state = pipe.read_json(self.root / "pipeline_state.json")
        self.assertEqual(state["stages"]["keys_smoke"]["returncode"], 9)
        with self.assertRaises(ValueError):
            self.execute(self.cli("full"))

    def test_empty_existing_directory_is_not_accepted_as_a_run(self):
        self.root.mkdir()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.execute(self.cli("full"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_changed_artifact_blocks_full_resume(self):
        self.execute()
        (self.root / "keys_smoke" / "paired_bank.pt").write_bytes(b"changed")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.execute(self.cli("full"))
        self.assertEqual(self.calls, [])

    def test_changed_protocol_or_code_blocks_full_resume(self):
        self.execute()
        self.calls.clear()
        with patch.object(pipe, "code_hash", return_value="new-code"):
            with self.assertRaisesRegex(ValueError, "completed smoke provenance"):
                pipe.new_or_resume(self.cli("full"), self.root)
        record = pipe.read_json(self.protocol)
        record["extra"] = "changed"
        self.save(self.protocol, record)
        with self.assertRaisesRegex(ValueError, "completed smoke provenance"):
            self.execute(self.cli("full"))
        self.assertEqual(self.calls, [])

    def test_changed_stage_log_blocks_resume(self):
        self.execute()
        (self.root / "logs" / "report_smoke.log").write_text("changed report")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.execute(self.cli("full"))
        self.assertEqual(self.calls, [])

    def test_full_never_overwrites_untracked_partial_outputs(self):
        self.execute()
        (self.root / "keys").mkdir()
        sentinel = self.root / "keys" / "keep.txt"
        sentinel.write_text("user partial evidence")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.execute(self.cli("full"))
        self.assertEqual(sentinel.read_text(), "user partial evidence")
        self.assertEqual(self.calls, [])

    def test_fake_full450_without_required_gate_is_rejected(self):
        self.execute()
        def failed_formal(command, log):
            code = self.child(command, log)
            if log.stem == "diagnose":
                for filename in ("summary.json", "result.json"):
                    path = self.root / "diagnose" / filename
                    record = pipe.read_json(path)
                    record["validation"]["labels"] = False
                    self.save(path, record)
            return code
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "evidence gates"):
            self.execute(self.cli("full"), failed_formal)
        self.assertEqual(self.calls, ["keys", "diagnose"])

    def test_diagnostic_bank_binding_is_checked(self):
        def wrong_bank(command, log):
            code = self.child(command, log)
            if log.stem == "diagnose_smoke":
                path = self.root / log.stem / "result.json"
                record = pipe.read_json(path)
                record["paired_bank_sha256"] = "other bank"
                self.save(path, record)
            return code
        with self.assertRaisesRegex(ValueError, "provenance binding"):
            self.execute(child=wrong_bank)
        self.assertEqual(self.calls, ["keys_smoke", "diagnose_smoke"])

    def test_plan_does_not_read_protocol_query_gpu_or_create_directories(self):
        cli = self.cli("plan", protocol=self.base / "missing_protocol.json")
        with patch.object(pipe, "read_json", side_effect=AssertionError("must not read")), \
                patch.object(pipe, "gpu_pids", side_effect=AssertionError("must not query")), \
                patch.object(pipe, "run_child", side_effect=AssertionError("must not run")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            pipe.run_pipeline(cli)
        self.assertFalse(self.root.exists())
        self.assertIn("--limit-images 4", out.getvalue())
        self.assertIn("--limit 2", out.getvalue())
        self.assertNotIn("train.py", out.getvalue())

    def test_busy_gpu_stops_before_creating_root(self):
        with patch.object(pipe, "code_hash", return_value="source-code-sha"), \
                patch.object(pipe, "gpu_pids", return_value=[12345]), \
                patch.object(pipe, "run_child", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(RuntimeError, "is busy"):
                pipe.run_pipeline(self.cli())
        self.assertFalse(self.root.exists())

    def test_gpu_wait_is_explicit_and_checks_periodically(self):
        cli = self.cli(wait_gpu_idle=True)
        with patch.object(pipe, "gpu_pids", side_effect=[[123], []]) as pids, \
                patch.object(pipe.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()) as out:
            pipe.ensure_idle("0", cli)
        self.assertEqual(pids.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertIn("Waiting for source GPU", out.getvalue())

    def test_gpu_wait_timeout_does_not_terminate_other_processes(self):
        cli = self.cli(wait_gpu_idle=True, gpu_wait_timeout=1)
        with patch.object(pipe, "gpu_pids", return_value=[123]), \
                patch.object(pipe.time, "monotonic", side_effect=[0, 0, 2]), \
                patch.object(pipe.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(TimeoutError, "waiting"):
                pipe.ensure_idle("0", cli)
        sleep.assert_called_once_with(1)

    def test_nonfinite_wait_timeout_and_symbolic_root_are_refused(self):
        for timeout in (float("nan"), float("inf"), -1):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                pipe.run_pipeline(self.cli(gpu_wait_timeout=timeout))
        destination = self.base / "elsewhere"
        destination.mkdir()
        self.root.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            pipe.run_pipeline(self.cli())
        self.assertEqual(list(destination.iterdir()), [])

    def test_failed_child_start_preserves_log_hash(self):
        def failed_start(command, log):
            log.write_text("Command written before failed process start")
            raise FileNotFoundError("missing executable")
        with self.assertRaises(FileNotFoundError):
            self.execute(child=failed_start)
        state = pipe.read_json(self.root / "pipeline_state.json")
        row = state["stages"]["keys_smoke"]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["returncode"])
        self.assertEqual(row["log_sha256"], pipe.digest(self.root / row["log"]))

    def test_device_comes_from_protocol_not_device_shell_variable(self):
        with patch.dict(os.environ, {"DEVICE": "cuda:7"}, clear=True):
            self.assertEqual(pipe.gpu_identifier({"arguments": {"device": "cuda:2"}}), "2")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5"}, clear=True):
            self.assertEqual(pipe.gpu_identifier({"arguments": {"device": "cuda:1"}}), "5")

    def test_report_requires_root_and_reads_only_completed_evidence(self):
        cli = self.cli("report", root=None)
        with self.assertRaisesRegex(ValueError, "requires --root"):
            pipe.run_pipeline(cli)
        self.execute()
        before = {path: pipe.digest(path) for path in self.root.rglob("*") if path.is_file()}
        with patch.object(pipe, "code_hash", return_value="source-code-sha"), \
                patch.object(pipe, "gpu_pids", side_effect=AssertionError("must not query")), \
                patch.object(pipe, "run_child", side_effect=AssertionError("must not run")), \
                contextlib.redirect_stdout(io.StringIO()):
            pipe.run_pipeline(self.cli("report"))
        self.assertEqual(before, {path: pipe.digest(path) for path in self.root.rglob("*") if path.is_file()})

    def test_shell_relocates_from_arbitrary_working_directory_and_ignores_key_roots(self):
        env = dict(os.environ, PYTHON=sys.executable, KEY_OUTPUT_ROOT="/bad/old/root", DIAG_OUTPUT="/bad/output", DEVICE="cuda:7")
        result = subprocess.run(["bash", str(pipe.ROOT / "run.sh"), "plan", "--root", str(self.root),
                                 "--protocol", str(self.protocol)], cwd=self.base, env=env,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root), result.stdout)
        self.assertIn(str(self.protocol), result.stdout)
        self.assertNotIn("/bad/old/root", result.stdout)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
