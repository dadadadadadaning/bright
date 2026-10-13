"""Relocation and explicit execution contracts of the unified entry point."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import RUN_TOOLS as tools


class NewToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo/New/moved_package"
        self.root.mkdir(parents=True)
        self.repo = self.root.parent.parent
        self.checkpoints = Path(self.temp.name) / "checkpoints"

    def cli(self, tool, **overrides):
        row = dict(tool=tool, repo=str(self.repo), checkpoints=str(self.checkpoints),
                   tools_root=None, protocol=None, evidence=None, run=False,
                   preflight=False, cpu_only=False)
        row.update(overrides)
        return SimpleNamespace(**row)

    def test_relocated_tools_keep_explicit_repository_and_checkpoint_sources(self):
        for name in ("memory", "memory-debug", "prt-debug", "prt-sequence", "region-night"):
            with self.subTest(name=name):
                command, env = tools.build_command(self.cli(name), self.root)
                self.assertTrue(any(str(self.root) in arg for arg in command))
                self.assertEqual(env["REPO"], str(self.repo))
                self.assertTrue(env["PYTHONPATH"].startswith(str(self.repo)))
                self.assertIn("--plan-only", command)
        command, _ = tools.build_command(self.cli("prt-sequence"), self.root)
        self.assertEqual(command[command.index("--night-tools") + 1], str(self.root / "region_night"))
        self.assertEqual(command[command.index("--failed-run") + 1], str(self.checkpoints / tools.FAILED_RUN))

    def test_actual_sequence_action_keeps_training_disabled(self):
        command, _ = tools.build_command(self.cli("prt-sequence", run=True), self.root)
        self.assertIn("--checks-only", command)
        self.assertIn("--wait-gpu-idle", command)
        self.assertNotIn("--continue-to-training", command)

    def test_memory_identity_action_preserves_explicit_evidence_and_cpu_only(self):
        evidence = self.root / "evidence.pt"
        command, _ = tools.build_command(self.cli("memory", run=True, cpu_only=True,
                                                 evidence=str(evidence)), self.root)
        self.assertIn("--cpu-only", command)
        self.assertEqual(command[command.index("--evidence") + 1], str(evidence))
        self.assertEqual(command[command.index("--tools-root") + 1], str(self.repo / "New"))

    def test_incompatible_actions_are_rejected(self):
        for cli in (self.cli("inventory", run=True), self.cli("region-night", preflight=True),
                    self.cli("prt-debug", cpu_only=True), self.cli("prt-sequence", evidence="evidence.pt"),
                    self.cli("prt-debug", protocol="protocol.json")):
            with self.assertRaises(ValueError):
                tools.build_command(cli, self.root)

    def test_checksum_mutation_is_rejected_before_child_execution(self):
        file = self.root / "program.py"
        file.write_bytes(b"original")
        (self.root / "RELEASE_SHA256.json").write_text(json.dumps({file.name: tools.digest(file)}))
        self.assertEqual(tools.verify_release(self.root), 1)
        file.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            tools.verify_release(self.root)

    def test_release_cannot_escape_package_or_follow_symbolic_links(self):
        outside = self.root.parent / "outside.py"
        outside.write_bytes(b"outside")
        (self.root / "link.py").symlink_to(outside)
        for name in ("../outside.py", "link.py", "/absolute.py"):
            with self.subTest(name=name):
                (self.root / "RELEASE_SHA256.json").write_text(json.dumps({name: tools.digest(outside)}))
                with self.assertRaises(ValueError):
                    tools.verify_release(self.root)

    def test_unified_plan_and_inventory_do_not_launch_any_process(self):
        with patch.object(tools, "ROOT", self.root), patch.object(tools, "verify_release", return_value=3), \
                patch.object(tools, "subprocess") as process, patch("builtins.print"):
            paths = ["--repo", str(self.repo), "--checkpoints", str(self.checkpoints)]
            self.assertEqual(tools.main(["--tool", "region-night", "--plan-only", *paths]), 0)
            self.assertEqual(tools.main(paths), 0)
            process.call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
