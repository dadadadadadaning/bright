"""Controller lock handoff and fail-visible startup contracts (no CUDA job)."""
import argparse
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import recovery

class LauncherContracts(unittest.TestCase):
    def test_controller_owns_lock_during_child_spawn_and_passes_descriptor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parent = root / 'parent'
            parent.mkdir()
            archive = root / 'example.pyz'
            archive.write_bytes(b'archive fixture')
            args = argparse.Namespace(parent=parent, package=root, output=None, lease_fd=None)
            def launch(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_LAUNCH_BLOCKING'], '1')
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(len(kwargs['pass_fds']), 1)
                self.assertIn(str(kwargs['pass_fds'][0]), command)
                with (root / '.readout_active.lock').open('a+') as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return SimpleNamespace(pid=12345)
            with patch.object(recovery, 'read_parent'), patch.object(recovery, 'gpu_idle'), \
                 patch.object(recovery.sys, 'argv', [str(archive)]), \
                 patch.object(recovery.subprocess, 'Popen', side_effect=launch) as spawn:
                recovery.start(args)
            self.assertEqual(spawn.call_count, 1)
            latest = recovery.read(root / '.read_aux_recovery_latest.json')
            output = Path(latest['output'])
            self.assertEqual((output / 'runner.pyz').read_bytes(), archive.read_bytes())
            self.assertEqual(recovery.read(output / 'summary.json')['phase'], 'starting')
            self.assertEqual(list(parent.iterdir()), [])

    def test_worker_lock_failure_writes_failed_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            parent = root / 'parent'; parent.mkdir()
            output = root / 'output'; output.mkdir()
            recovery.write(output / 'summary.json', {'phase': 'starting'})
            args = argparse.Namespace(parent=parent, output=output, lease_fd=None)
            with (root / '.readout_active.lock').open('a+') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.dict(recovery.os.environ, CUDA_LAUNCH_BLOCKING='1'), \
                     patch.object(recovery, 'run_locked') as worker:
                    with self.assertRaises(BlockingIOError):
                        recovery.run_worker(args)
            self.assertEqual(worker.call_count, 0)
            self.assertEqual(recovery.read(output / 'summary.json')['phase'], 'failed')

if __name__ == '__main__':
    unittest.main()
