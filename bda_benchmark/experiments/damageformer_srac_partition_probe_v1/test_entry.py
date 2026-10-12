"""Synthetic CPU end-to-end tests; not a replacement for BRIGHT/CUDA smoke."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.test_entry import TinyFront
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SRACTransformerFront
from . import audit


class ProbeEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.flags = (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        self.workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')

    def tearDown(self):
        enabled, warn, deterministic, benchmark = self.flags
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if self.workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = self.workspace

    def fixture(self, output, suite='all', smoke=True):
        args = audit.parser().parse_args([
            '--checkpoint', 'not-opened.pth', '--a1_checkpoint', 'not-opened-a1.pth',
            '--dataset', 'not-opened', '--train_list', 'train_set.txt', '--val_list', 'val_set.txt',
            '--output', str(output), '--suite', suite, '--device', 'cpu', '--workers', '0',
        ] + (['--smoke'] if smoke else []))
        torch.manual_seed(117)
        front = TinyFront().requires_grad_(False).eval()
        front.partition.stride = 8
        front.partition.iterations = 3
        pre, post = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        target = torch.arange(32*32).reshape(1, 32, 32) % 4
        target[:, :2, :2] = 255
        batches = [(pre, post, (target > 0).long(), target, ['Beirut_image' + str(i)]) for i in range(2)]
        payload = dict(load_report={'expected_valid_miou': .61}, valid_miou=.60)
        return args, front, payload, batches

    def run_mock(self, args, front, payload, batches, load_effect=None):
        with mock.patch.object(audit, 'load_fixed', return_value=(front, payload, {}), side_effect=load_effect), \
             mock.patch.object(audit, 'lists', return_value=([], [f'Beirut_{i}' for i in range(450)])), \
             mock.patch.object(audit, 'loader', return_value=batches), \
             mock.patch.object(audit, 'parse_batch', side_effect=lambda batch, device: batch), \
             contextlib.redirect_stdout(io.StringIO()):
            audit.run(args)

    def test_all_suites_write_serializable_evidence_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'probe'
            args, front, payload, batches = self.fixture(output)
            before = module_fingerprint(front)
            self.run_mock(args, front, payload, batches)
            self.assertEqual(before, module_fingerprint(front))
            self.assertTrue(all(p.grad is None for p in front.parameters()))
            status = json.loads((output/'summary.json').read_text())
            self.assertTrue(status['complete'])
            self.assertFalse(status['full_validation_evidence'])
            self.assertFalse(status['test_accessed'])
            self.assertTrue(status['execution']['deterministic_algorithms'])
            self.assertFalse(status['execution']['deterministic_warn_only'])
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['execution'], status['execution'])
            self.assertEqual(manifest['q_replay_tolerance'], 2e-5)
            result = json.loads((output/'audit.json').read_text())
            self.assertIn('b_bounded', result['metrics'])
            self.assertIn('radius2_iter3_probq', result['metrics'])
            self.assertIn('edge_change', result['metrics'])
            self.assertIn('roi_false_negative', result['groups'])
            self.assertEqual(result['routing']['radius2_iter3']['image_means']['supports'], 25)
            rows = [json.loads(line) for line in (output/'per_image.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]['iteration_trace']['iter5']), 5)
            self.assertLess(rows[0]['q_replay_max_abs'], 2e-5)

    def test_runtime_failure_is_saved_without_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'unsupported'
            args, front, payload, batches = self.fixture(output)
            with mock.patch.object(audit, 'configure_determinism', side_effect=RuntimeError('unsupported kernel')), \
                 mock.patch.object(audit, 'capture_inputs') as capture_mock:
                with self.assertRaisesRegex(RuntimeError, 'unsupported kernel'):
                    self.run_mock(args, front, payload, batches)
                capture_mock.assert_not_called()
            status = json.loads((output/'summary.json').read_text())
            self.assertFalse(status['complete'])
            self.assertIn('unsupported kernel', status['failure']['message'])

    def test_loading_cannot_silently_weaken_runtime_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'weakened'
            args, front, payload, batches = self.fixture(output)
            def weaken(_args):
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                torch.use_deterministic_algorithms(True, warn_only=True)
                return front, payload, {}
            with self.assertRaisesRegex(RuntimeError, 'Strict deterministic execution'):
                self.run_mock(args, front, payload, batches, load_effect=weaken)
            status = json.loads((output/'summary.json').read_text())
            self.assertFalse(status['complete'])

    def test_failed_replay_remains_incomplete_and_saves_context(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'failed'
            args, front, payload, batches = self.fixture(output, 'prior')
            original = audit.partition_prediction
            def bad(*a, **kw):
                probability = original(*a, **kw)
                return probability.roll(1, 1)
            with mock.patch.object(audit, 'partition_prediction', side_effect=bad):
                with self.assertRaisesRegex(RuntimeError, 'replay failed'):
                    self.run_mock(args, front, payload, batches)
            status = json.loads((output/'summary.json').read_text())
            self.assertFalse(status['complete'])
            self.assertEqual(status['failure']['image'], 'Beirut_image0')

    def test_full_validation_gate_cannot_be_satisfied_by_two_images(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'not_full'
            args, front, payload, batches = self.fixture(output, 'prior', smoke=False)
            with self.assertRaisesRegex(RuntimeError, 'gate FAILED'):
                self.run_mock(args, front, payload, batches)
            summary = json.loads((output/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['reproduction']['passed'])
            self.assertTrue((output/'audit.json').is_file())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            args, front, payload, batches = self.fixture(temp)
            with self.assertRaises(FileExistsError):
                self.run_mock(args, front, payload, batches)

    def test_boundary_predictions_use_captured_q_not_replay_q(self):
        with tempfile.TemporaryDirectory() as temp:
            args, front, payload, batches = self.fixture(Path(temp)/'boundary', 'boundary')
            original_capture, original_prediction = audit.capture_inputs, audit.partition_prediction
            captured = {}
            edge_calls = []
            def capture_once(*a, **kw):
                native, fixed = original_capture(*a, **kw)
                captured['q_pointer'] = native['q'].data_ptr()
                return native, fixed
            def check_q(model, fixed, part, size, adapted=None):
                if adapted is not None:
                    self.assertEqual(part['q'].data_ptr(), captured['q_pointer'])
                    edge_calls.append(True)
                return original_prediction(model, fixed, part, size, adapted=adapted)
            with mock.patch.object(audit, 'capture_inputs', side_effect=capture_once), \
                 mock.patch.object(audit, 'partition_prediction', side_effect=check_q):
                self.run_mock(args, front, payload, batches)
            self.assertEqual(len(edge_calls), 6)

    def test_byte_checks_detect_inplace_modification_and_nonfinite(self):
        before = torch.arange(8, dtype=torch.float32)
        after = before.clone()
        self.assertTrue(audit.compare_tensor(after, before)['passed'])
        after[3] += 1
        self.assertEqual(audit.compare_tensor(after, before)['changed_elements'], 1)
        after[4] = float('nan')
        self.assertFalse(audit.compare_tensor(after, before)['passed'])

    def test_real_swin_front_capture_and_legacy_replay(self):
        torch.manual_seed(52)
        front = SRACTransformerFront(dim=16, decoder_width=16, stride=8, iterations=3)
        front.requires_grad_(False).eval()
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        before = module_fingerprint(front)
        with torch.no_grad():
            native, fixed = audit.capture_inputs(front, pre, post)
            part = audit.run_partition(front.partition, fixed['structural'], fixed['B'])
            result = audit.partition_prediction(front, fixed, part, (64, 64))
        torch.testing.assert_close(part['q'], native['q'], atol=0, rtol=0)
        torch.testing.assert_close(result, native['prob'], atol=2e-5, rtol=0)
        self.assertEqual(part['shape'], (16, 16))
        self.assertEqual(module_fingerprint(front), before)


if __name__ == '__main__':
    unittest.main()
