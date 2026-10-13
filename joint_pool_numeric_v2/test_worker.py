import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from pool_ops import tensor_layout
import worker


def snapshot():
    x = torch.tensor([1.0, -2.0, 3.0, 0.5, 0.0, 4.0]).reshape(1, 1, 2, 3)
    q = torch.arange(1, 55, dtype=torch.float32).reshape(1, 9, 6) / 55
    ids = torch.arange(54).reshape(9, 6).remainder(4)
    return dict(format='native_pool_input_v1', content=x, weights=q, ids=ids,
                token_count=4, image='synthetic', call=2,
                layout={k: tensor_layout(v) for k, v in zip(('content', 'weights', 'ids'), (x, q, ids))})


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.pth'
        torch.save(snapshot(), self.source)
        self.args = argparse.Namespace(backend='cpu', input=str(self.source),
            output=str(self.root / 'cpu'), expected_sha256=worker.sha256(self.source),
            expected_image='synthetic', expected_call=2, runtime_json=None,
            repeats=3, device='cuda:0', audit_root=str(self.root))

    def test_cpu_real_worker_without_cuda(self):
        with mock.patch('torch.cuda.is_available', side_effect=AssertionError('CUDA probed')):
            record = worker.run(self.args)
        self.assertTrue(record['execution_complete'])
        self.assertTrue(record['finite_outputs'])
        self.assertFalse(record['training_allowed'])
        self.assertTrue(record['comparisons']['cpu64_scatter_vs_grouped']['passed'])
        outputs = torch.load(self.root / 'cpu/outputs.pt', weights_only=True)
        self.assertEqual(outputs['cpu32']['tokens'].dtype, torch.float32)
        self.assertEqual(outputs['cpu64']['tokens'].dtype, torch.float64)
        self.assertEqual(record['input_sha256'], worker.sha256(self.source))

    def test_input_hash_failure_is_recorded(self):
        self.args.expected_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            worker.run(self.args)
        record = json.loads((self.root / 'cpu/summary.json').read_text())
        self.assertFalse(record['execution_complete'])
        self.assertEqual(record['failure']['phase'], 'load_saved_input')

    def test_image_and_call_binding(self):
        for name, value in [('expected_image', 'wrong'), ('expected_call', 3)]:
            args = copy.copy(self.args)
            setattr(args, name, value)
            with self.assertRaises(ValueError):
                worker.load_bound_snapshot(args)

    def test_existing_output_refused(self):
        (self.root / 'cpu').mkdir()
        sentinel = self.root / 'cpu/keep'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            worker.run(self.args)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_partial_compare_never_full_diagnostic(self):
        worker.run(self.args)
        self.args.backend = 'compare'
        self.args.output = str(self.root / 'compare')
        with mock.patch('builtins.print'):
            record = worker.run(self.args)
        self.assertTrue(record['execution_complete'])
        self.assertFalse(record['diagnostic_complete'])
        self.assertEqual(record['missing_workers'], ['native', 'contiguous'])
        self.assertFalse(record['training_allowed'])

    def test_localize_valid_and_invalid_tokens(self):
        ss = snapshot()
        reference = dict(tokens=torch.zeros(1, 3, 2), mass=torch.tensor([[1., 0., 3.]]))
        actual = dict(tokens=torch.zeros(1, 3, 2), mass=reference['mass'].clone())
        actual['tokens'][0, 0, 0] = 1
        actual['tokens'][0, 1, 0] = 2
        actual['mass'][0, 1] = 1
        details = worker.localize_tokens(actual, reference, ss)
        self.assertEqual(details['distinct_tokens_outside_tolerance'], 2)
        self.assertEqual(details['valid_token_elements_outside'], 1)
        self.assertEqual(details['invalid_token_elements_outside'], 1)
        self.assertEqual(details['validity_disagreements'], 1)
        self.assertFalse(details['saved_input_is_raw_and_fp32_square'])

    def test_derived_stats_only_with_saved_square_evidence(self):
        ss = snapshot()
        raw = torch.arange(256, dtype=torch.float32).reshape(1, 256, 1, 1) / 100
        ss['content'] = torch.cat([raw, raw.square()], 1)
        toks = ss['content'].flatten(2).transpose(1, 2)
        outputs = dict(tokens=toks, mass=torch.ones(1, 1))
        detail = worker.localize_tokens(outputs, outputs, ss)
        self.assertTrue(detail['saved_input_is_raw_and_fp32_square'])
        self.assertEqual(detail['derived_std']['max_abs'], 0)
        ss['content'][0, -1] += 1
        self.assertNotIn('derived_std', worker.localize_tokens(outputs, outputs, ss))

    def test_classification_requires_repeat_stability(self):
        cpu = dict(execution_complete=True, comparisons={'cpu64_scatter_vs_grouped': {'passed': True}})
        row = dict(original_cpu32_gate={'passed': False}, vs_cpu64={'passed': True}, vs_first={'passed': True})
        gpu = dict(execution_complete=True, repeats=[row])
        message = worker.classify(cpu, gpu, gpu, {'passed': True})
        self.assertIn('supports', message)
        self.assertIn('does not validate', message)
        row['vs_first']['passed'] = False
        self.assertIn('not repeat-stable', worker.classify(cpu, gpu, gpu, {'passed': True}))


if __name__ == '__main__':
    unittest.main()
