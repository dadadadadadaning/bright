"""Exercise the real background launcher with a tiny, Torch-free ZIP worker.

Only preflight is mocked. These tests cover process/lease/archive behavior,
not the CUDA training worker or its scientific results.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import resume_train


WORKER = """\
import hashlib
import json
import os
from pathlib import Path
import sys
import time

assert sys.argv[1] == "worker"
output = Path(sys.argv[sys.argv.index("--output") + 1])
descriptor = int(os.environ["BRIGHT_RESUME_LOCK_FD"])
os.fstat(descriptor)
archive = Path(sys.argv[0]).resolve()
(output / "ready.json").write_text(json.dumps({
    "pid": os.getpid(), "archive": str(archive),
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "torch_imported": "torch" in sys.modules,
}))
deadline = time.monotonic() + 1.75
while not (output / "release").is_file():
    if time.monotonic() >= deadline:
        raise SystemExit("test handshake timed out")
    time.sleep(0.01)
"""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.original = self.folder / "original"
        self.original.mkdir()
        (self.original / "checkpoint_001000.pt").write_bytes(b"original checkpoint")
        (self.original / "train.jsonl").write_text('{"fixture": true}\n')
        self.artifact = self.folder / "release.pyz"
        with zipfile.ZipFile(self.artifact, "w") as archive:
            archive.writestr("__main__.py", WORKER)
        self.original_snapshot = self.snapshot_original()
        self.output_base = self.folder / "continuations"
        self.context = dict(
            root=self.original, addon=self.original / "addon", seed=3407,
            checkpoint_every=50, lineage={"fixture": "launch only"},
            parent=dict(kind="original", path=self.original / "checkpoint_001000.pt",
                        step=1000, sha256="fixture"),
            evidence_files={},
            plan={"source": {
                "package": str(self.original / "package"),
                "source_run": str(self.original / "source_run"),
                "config": {
                    "repo": str(self.original / "repo"),
                    "bank_root": str(self.original / "banks"),
                    "values": str(self.original / "values"),
                },
            }},
        )
        self.args = SimpleNamespace(output_base=str(self.output_base), parent_root=None)
        self.processes = []
        # Register after temporary cleanup so children are reaped first.
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

    def snapshot_original(self):
        return {
            str(path.relative_to(self.original)):
                (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in self.original.rglob("*") if path.is_file()
        }

    def wait_ready(self, output, process):
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            path = output / "ready.json"
            if path.is_file():
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            if process.poll() is not None:
                self.fail("Tiny worker exited before handshake: " +
                          (output / "train.log").read_text())
            time.sleep(0.01)
        self.fail("Tiny worker did not become ready within 1.5 seconds")

    def test_pinned_archive_and_inherited_lease_survive_launcher_return(self):
        original_bytes = self.artifact.read_bytes()
        original_digest = hashlib.sha256(original_bytes).hexdigest()
        actual_popen = subprocess.Popen

        def start(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            self.processes.append(process)
            return process

        with (
            mock.patch.object(resume_train, "prepare", return_value=self.context),
            mock.patch.object(resume_train.sys, "argv", [str(self.artifact)]),
            mock.patch.object(resume_train.subprocess, "Popen", side_effect=start),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(resume_train.launch(self.args), 0)
            self.assertEqual(len(self.processes), 1)
            output = Path((self.output_base / "latest.txt").read_text().strip())
            process = self.processes[0]

            # A later release replacement must not alter the launched worker.
            self.artifact.write_bytes(b"replacement that is not an executable ZIP")
            ready = self.wait_ready(output, process)
            self.assertEqual(ready["archive"], str(output / "trainer.pyz"))
            self.assertEqual(ready["sha256"], original_digest)
            self.assertFalse(ready["torch_imported"])
            self.assertEqual((output / "trainer.pyz").read_bytes(), original_bytes)
            request = json.loads((output / "request.json").read_text())
            self.assertEqual(request["trainer_sha256"], original_digest)
            self.assertEqual(ready["pid"], process.pid)
            self.assertIsNone(process.poll())

            with (self.output_base / "active.lock").open("a+") as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    resume_train.launch(self.args)
                self.assertEqual(len(self.processes), 1)
                self.assertEqual(list(self.output_base.glob("resume_*")), [output])

                (output / "release").touch()
                self.assertEqual(process.wait(timeout=1.5), 0)
                # The worker exit releases the inherited lease automatically.
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(contender, fcntl.LOCK_UN)

        self.assertEqual(self.snapshot_original(), self.original_snapshot)

    def test_original_tree_cannot_be_the_background_output(self):
        self.args.output_base = str(self.original / "accidental_output")
        with (
            mock.patch.object(resume_train, "prepare", return_value=self.context),
            mock.patch.object(resume_train.sys, "argv", [str(self.artifact)]),
            mock.patch.object(resume_train.subprocess, "Popen") as popen,
        ):
            with self.assertRaisesRegex(ValueError, "outside original"):
                resume_train.launch(self.args)
        popen.assert_not_called()
        self.assertFalse((self.original / "accidental_output").exists())
        self.assertEqual(self.snapshot_original(), self.original_snapshot)


if __name__ == "__main__":
    unittest.main()
