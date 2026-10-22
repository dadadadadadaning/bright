import contextlib
import gc
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import bright_tensor_lifetime_v1 as probe


class LifetimeTests(unittest.TestCase):
    def test_local_scope_releases_cpu_tensors_and_does_not_train(self):
        before = torch.is_grad_enabled()
        refs = probe.one_round(torch, 'cpu', elements=128)
        self.assertEqual(len(refs), 3)
        self.assertTrue(all(r() is None for r in refs))
        self.assertEqual(torch.is_grad_enabled(), before)

    def test_retained_frame_is_visible_to_weakrefs(self):
        frames = []
        old = sys.getprofile()
        def profile(frame, event, arg):
            if event == 'return' and frame.f_code is probe.one_round.__code__:
                frames.append(frame)
        try:
            sys.setprofile(profile)
            refs = probe.one_round(torch, 'cpu', elements=128)
        finally:
            sys.setprofile(old)
        self.assertEqual(sum(r() is not None for r in refs), 3)
        frames.clear()
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))

    def test_commands_share_same_script_and_use_memcheck_only_in_second_case(self):
        root = Path('/tmp/probe')
        plain = probe.command(root, 'plain', {'sanitizer': '/tool'})
        mem = probe.command(root, 'memcheck', {'sanitizer': '/tool'})
        self.assertEqual(plain[:-1], mem[-len(plain):-1])
        self.assertEqual(plain[-1], 'plain')
        self.assertEqual(mem[-1], 'memcheck')
        self.assertEqual(mem[:5], ['/tool', '--tool', 'memcheck', '--error-exitcode', '86'])

    def test_fresh_interpreter_import_does_not_import_torch_or_touch_cuda(self):
        code = 'import sys; import bright_tensor_lifetime_v1; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).parent, check=True)

    def test_supervisor_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(probe, 'command', return_value=[sys.executable, '-c', 'import time; time.sleep(30)']), contextlib.redirect_stdout(io.StringIO()):
                result = probe.supervise(root, 'plain', {'timeout_per_case_seconds': 0.1}, {'requested': False})
            self.assertEqual(result['supervisor_reason'], 'time_budget_exceeded')
            self.assertFalse(result['accepted'])
            self.assertNotEqual(result['returncode'], 0)

    def test_zero_child_exit_without_sanitizer_summary_is_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = str(root / 'memcheck/summary.json')
            code = 'from pathlib import Path; Path(%r).write_text(\'{"complete":true}\')' % out
            with mock.patch.object(probe, 'command', return_value=[sys.executable, '-c', code]), contextlib.redirect_stdout(io.StringIO()):
                result = probe.supervise(root, 'memcheck', {'timeout_per_case_seconds': 5}, {'requested': False})
            self.assertEqual(result['returncode'], 0)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['sanitizer_error_counts'], [])

    def test_plain_failure_prevents_second_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(probe, 'leases', return_value=contextlib.nullcontext()), \
                 mock.patch.object(probe, 'read_request', return_value={}), \
                 mock.patch.object(probe, 'supervise', return_value={'accepted': False}) as supervise, \
                 mock.patch.object(probe.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.worker(root), 1)
            self.assertEqual(supervise.call_count, 1)
            s = json.loads((root / 'summary.json').read_text())
            self.assertTrue(s['job_finished'])
            self.assertFalse(s['historical_failure_resolved'])
            self.assertFalse(s['test_accessed'])

    def test_success_does_not_clear_hardware_or_original_fault(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(probe, 'leases', return_value=contextlib.nullcontext()), \
                 mock.patch.object(probe, 'read_request', return_value={}), \
                 mock.patch.object(probe, 'supervise', return_value={'accepted': True}), \
                 mock.patch.object(probe.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.worker(root), 0)
            s = json.loads((root / 'summary.json').read_text())
            self.assertTrue(s['diagnostic_complete'])
            for key in ('hardware_cleared', 'hardware_fault_proven', 'historical_failure_resolved', 'test_accessed', 'training_performed'):
                self.assertFalse(s[key])


if __name__ == '__main__':
    unittest.main()
