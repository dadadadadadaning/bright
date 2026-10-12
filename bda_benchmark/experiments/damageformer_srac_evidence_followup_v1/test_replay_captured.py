"""Fresh-process replay must retain failures and still test other backends."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import replay_captured


class CapturedReplayContracts(unittest.TestCase):
    def test_failed_native_does_not_skip_grouped_or_claim_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root/'capture'/'pool_debug'
            capture.mkdir(parents=True)
            (capture/'input_0001.pth').write_bytes(b'opaque trusted input')
            calls = []

            def child(cmd, **kwargs):
                backend = cmd[cmd.index('--backend')+1]
                output = Path(cmd[cmd.index('--output')+1])
                output.mkdir()
                good = backend != 'native'
                (output/'summary.json').write_text(json.dumps({'gates_passed': good}))
                self.assertEqual(kwargs['env']['CUDA_LAUNCH_BLOCKING'], '1')
                calls.append(backend)
                return SimpleNamespace(returncode=0 if good else 1)

            args = argparse.Namespace(capture_run=str(capture.parent), output=str(root/'replay'),
                                      device='cuda:0', repeats=3, backends=['cpu', 'native', 'grouped'])
            with patch.object(replay_captured.subprocess, 'run', side_effect=child):
                result = replay_captured.run(args)
            self.assertEqual(calls, ['cpu', 'native', 'grouped'])
            self.assertTrue(result['complete'])
            self.assertFalse(result['gates_passed'])
            self.assertFalse(result['full_validation_evidence'])
            self.assertEqual(result['backend_passed'], {'cpu': True, 'native': False, 'grouped': True})
            self.assertEqual((capture/'input_0001.pth').read_bytes(), b'opaque trusted input')

    def test_no_inputs_or_nested_output_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(capture_run=str(root), output=str(root/'nested'),
                                      device='cpu', repeats=1, backends=['cpu'])
            with self.assertRaisesRegex(ValueError, 'Requires 1'):
                replay_captured.run(args)
            (root/'input_0001.pth').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'outside'):
                replay_captured.run(args)
            self.assertFalse((root/'nested').exists())


if __name__ == '__main__':
    unittest.main()
