"""CPU routing checks for the offline New-layout Memory launcher."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("new_memory_launcher", HERE / "RUN_MEMORY_REPLAY.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bright-new-memory-test-")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "BRIGHT-master"
        (self.repo / "bda_benchmark").mkdir(parents=True)
        self.tools = self.repo / "New"
        self.tools.mkdir()
        self.checkpoints = self.root / "BRIGHT-checkpoints"
        self.checkpoints.mkdir()
        self.payload = b"fixed fixture evidence; the real SHA remains pinned in production"
        probe_fixture = self.root / "fixture.pt"
        probe_fixture.write_bytes(self.payload)
        self.evidence_sha = launcher.digest(probe_fixture)
        self.sha_patch = patch.object(launcher, "EVIDENCE_SHA256", self.evidence_sha)
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def stage(self, parent, name):
        stage = parent / name / "keys_smoke"
        evidence = stage / "replay_evidence/00003.pt"
        evidence.parent.mkdir(parents=True)
        evidence.write_bytes(self.payload)
        record = dict(format="SRAC_MEMORY_PAIRED_KEYS_V2", complete=False, smoke=True,
                      last_replay_evidence=dict(path="replay_evidence/00003.pt", sha256=self.evidence_sha))
        (stage / "summary.json").write_text(json.dumps(record), encoding="utf-8")
        return evidence

    def protocol(self, parent, name="protocol.json"):
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(format="original-protocol-fixture", input_paths=dict(bank="/old/unchanged/bank.pt"))
        path.write_text(json.dumps(record), encoding="utf-8")
        return path, launcher.digest(path)

    def test_searches_new_and_checkpoints_and_deduplicates_same_canonical_path(self):
        evidence = self.stage(self.tools, "BRIGHT-memory-debug_esfI2YOk")
        found, summaries = launcher.find_evidence((self.checkpoints, self.tools, self.repo))
        self.assertEqual(found, evidence.resolve())
        self.assertEqual(summaries, [evidence.parent.parent / "summary.json"])
        # The nested New tree and its parent see the same artifact, not two runs.
        self.assertEqual(len(summaries), 1)

    def test_duplicate_results_refuse_latest_but_allow_explicit_sha_bound_selection(self):
        first = self.stage(self.checkpoints, "run_old")
        second = self.stage(self.tools, "run_newer")
        with self.assertRaisesRegex(ValueError, "Multiple distinct"):
            launcher.find_evidence((self.checkpoints, self.tools))
        found, summaries = launcher.find_evidence((self.checkpoints, self.tools), str(first))
        self.assertEqual(found, first.resolve())
        self.assertEqual(summaries, [first.parent.parent / "summary.json"])
        found, _ = launcher.find_evidence((self.checkpoints, self.tools), second)
        self.assertEqual(found, second.resolve())

    def test_explicit_selection_cannot_bypass_summary_path_or_v2_format(self):
        evidence = self.stage(self.tools, "run")
        other = evidence.with_name("other.pt")
        other.write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, "does not match its keys_smoke summary"):
            launcher.find_evidence((self.tools,), other)
        summary = evidence.parent.parent / "summary.json"
        record = json.loads(summary.read_text())
        record["format"] = "wrong-version"
        summary.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "non-V2"):
            launcher.find_evidence((self.tools,), evidence)

    def test_matching_summary_with_tampered_evidence_is_not_silently_skipped(self):
        evidence = self.stage(self.checkpoints, "run")
        evidence.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "has changed"):
            launcher.find_evidence((self.checkpoints,))
        with self.assertRaisesRegex(ValueError, "has changed"):
            launcher.find_evidence((self.checkpoints,), evidence)

    def test_relative_escape_and_absolute_evidence_paths_are_rejected(self):
        evidence = self.stage(self.checkpoints, "run")
        summary = evidence.parent.parent / "summary.json"
        for relative in ("../../outside.pt", str(evidence)):
            with self.subTest(relative=relative):
                record = json.loads(summary.read_text())
                record["last_replay_evidence"]["path"] = relative
                summary.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "escapes|relative"):
                    launcher.find_evidence((self.checkpoints,))

    def test_protocol_identity_uses_original_bytes_and_refuses_wrong_explicit_file(self):
        protocol, sha = self.protocol(self.checkpoints / launcher.DEFAULT_PROTOCOL.parent)
        before = protocol.read_bytes()
        self.assertEqual(launcher.find_protocol(self.checkpoints, self.tools, None, sha), protocol.resolve())
        self.assertEqual(launcher.find_protocol(self.checkpoints, self.tools, protocol.parent, sha), protocol.resolve())
        wrong = self.tools / "wrong.json"
        wrong.write_text("{}")
        with self.assertRaisesRegex(ValueError, "does not match"):
            launcher.find_protocol(self.checkpoints, self.tools, wrong, sha)
        self.assertEqual(protocol.read_bytes(), before)
        self.assertEqual(json.loads(protocol.read_text())["input_paths"]["bank"], "/old/unchanged/bank.pt")

    def test_fallback_protocol_duplicates_require_explicit_selection(self):
        first, sha = self.protocol(self.checkpoints / "unknown_run")
        second = self.tools / "copied/protocol.json"
        second.parent.mkdir()
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, "found 2"):
            launcher.find_protocol(self.checkpoints, self.tools, None, sha)
        self.assertEqual(launcher.find_protocol(self.checkpoints, self.tools, first, sha), first.resolve())

    def test_installed_source_sha_binding_is_checked_without_modifying_shared(self):
        shared = self.repo / "bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/shared.py"
        shared.parent.mkdir(parents=True)
        expected = "7" * 64
        shared.write_text("def source_hash():\n    return " + repr(expected) + "\n")
        before = shared.read_bytes()
        self.assertEqual(launcher.check_v2_source(self.repo, expected), expected)
        with self.assertRaisesRegex(ValueError, "differs"):
            launcher.check_v2_source(self.repo, "8" * 64)
        self.assertEqual(shared.read_bytes(), before)
        self.assertFalse((shared.parent / "__pycache__").exists())

    def test_default_plan_is_offline_no_forward_no_output_and_cpu_run_delegates(self):
        evidence = self.stage(self.checkpoints, "run")
        protocol, protocol_sha = self.protocol(self.checkpoints / launcher.DEFAULT_PROTOCOL.parent)
        package = self.tools / "new_package"
        package.mkdir()
        probe = package / launcher.PROBE_NAME
        probe.write_text("original fixture probe")
        output = self.checkpoints / "new_report"
        identity = dict(protocol_sha256=protocol_sha, source_sha256="9" * 64, image="train-only-image")
        argv = ["--repo", str(self.repo), "--checkpoints", str(self.checkpoints), "--output", str(output)]
        with patch.object(launcher, "__file__", str(package / "RUN_MEMORY_REPLAY.py")), \
                patch.object(launcher, "PROBE_SHA256", launcher.digest(probe)), \
                patch.object(launcher, "load_evidence_identity", return_value=identity), \
                patch.object(launcher, "check_v2_source", return_value=identity["source_sha256"]), \
                patch.object(launcher.subprocess, "call", return_value=17) as call, \
                redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(launcher.main(argv), 0)
            call.assert_not_called()
            plan = json.loads(stdout.getvalue())
            self.assertTrue(plan["offline"])
            self.assertFalse(plan["training"] or plan["test_accessed"] or plan["gpu_ran"])
            self.assertEqual(plan["evidence"], str(evidence))
            self.assertEqual(plan["protocol"], str(protocol))
            self.assertEqual(plan["model_forward_count"], 0)
            self.assertEqual(len(plan["known_tool_directories"]), 5)
            self.assertFalse(output.exists())
            self.assertEqual(launcher.main(argv + ["--run", "--cpu-only"]), 17)
            command = call.call_args.args[0]
            self.assertIn("--cpu-only", command)
            self.assertEqual(command[1], str(probe))
            self.assertFalse(output.exists())  # delegated fixture did not create a report

    def test_fixed_artifact_is_safely_loaded_on_cpu_with_real_torch(self):
        import torch
        evidence = self.root / "safe_evidence.pt"
        data = dict(format="SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2", protocol_sha256="a" * 64,
                    source_sha256="b" * 64, replayed_fp32=torch.tensor([0.25], dtype=torch.float32))
        torch.save(data, evidence)
        original_load = torch.load
        with patch.object(torch, "load", wraps=original_load) as load:
            result = launcher.load_evidence_identity(evidence)
        self.assertEqual(load.call_args.kwargs, dict(map_location="cpu", weights_only=True))
        self.assertEqual(result["replayed_fp32"].device.type, "cpu")
        self.assertEqual(result["protocol_sha256"], data["protocol_sha256"])


if __name__ == "__main__":
    unittest.main()
