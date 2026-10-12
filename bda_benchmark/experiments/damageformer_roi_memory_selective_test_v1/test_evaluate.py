"""Small CPU checks for the one-time Test protocol's failure boundaries."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from . import evaluate
from bda_benchmark.experiments.damageformer_roi_memory_selective_v1.memory import (
    HEAD_FORMAT, POSTERIOR_MODES, SelectivePosteriorHead,
)


class TestProtocol(unittest.TestCase):
    def test_disjoint_standard_splits_and_missing_targets_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / 'standard_ML'
            parent.mkdir()
            for file, prefix, count in (
                    ('train_set.txt', 'train_', 2918),
                    ('val_set.txt', 'val_', 450),
                    ('test_set.txt', 'test_', 878)):
                (parent / file).write_text(
                    ''.join(f'{prefix}{number}\n' for number in range(count)),
                    encoding='utf-8')
            test_path, names = evaluate.resolve_test_names(
                parent / 'train_set.txt', parent / 'val_set.txt')
            self.assertEqual(test_path.name, 'test_set.txt')
            self.assertEqual(len(names), 878)
            with self.assertRaisesRegex(FileNotFoundError, 'No Test mIoU'):
                evaluate.check_test_files(Path(temp) / 'dataset', names)
            (parent / 'test_set.txt').write_text(
                'train_0\n' + ''.join(
                    f'test_{number}\n' for number in range(1, 878)),
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                evaluate.resolve_test_names(
                    parent / 'train_set.txt', parent / 'val_set.txt')

    def test_only_completed_final_last_heads_and_fixed_metrics_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exemplar = SelectivePosteriorHead(8)
            count = sum(value.numel() for value in exemplar.parameters())
            metrics = {name: {'valid_miou': .4}
                       for name in POSTERIOR_MODES}
            manifest = dict(code_hash='frozen_code',
                            identity={'source': 'Train'},
                            bank_file_sha256={'bank': 'fixed'},
                            head_parameters={name: count
                                             for name in POSTERIOR_MODES})
            summary = dict(final_step=5000, final_metrics=metrics)
            paths = {}
            for name in POSTERIOR_MODES:
                path = root / (name + '_last.pth')
                paths[name] = path
                torch.save(dict(format=HEAD_FORMAT, stage='posterior',
                                name=name, step=5000,
                                code_hash='frozen_code',
                                source_identity=manifest['identity'],
                                bank_file_sha256=manifest['bank_file_sha256'],
                                metrics=metrics[name], head=exemplar.state_dict()), path)
            heads, hashes = evaluate.load_heads(
                paths, manifest, summary, 8, torch.device('cpu'))
            self.assertEqual(set(heads), set(POSTERIOR_MODES))
            self.assertEqual(len(hashes), 4)
            self.assertTrue(all(not parameter.requires_grad
                                for head in heads.values()
                                for parameter in head.parameters()))
            bad = torch.load(paths['posterior_nogate'], map_location='cpu',
                             weights_only=False)
            bad['step'] = 4500
            torch.save(bad, paths['posterior_nogate'])
            with self.assertRaisesRegex(ValueError, 'identity/metrics mismatch'):
                evaluate.load_heads(paths, manifest, summary, 8,
                                    torch.device('cpu'))

    def test_explicit_test_access_gate_precedes_everything(self):
        with mock.patch.dict('os.environ', {'ALLOW_TEST_ACCESS': ''}):
            with self.assertRaises(PermissionError):
                evaluate.run('missing', 'missing', 'cpu', 0)

    def test_val_replay_blocks_wrong_final_head_before_test(self):
        names = evaluate.PREDICTIONS
        confusion = [[100, 0, 0, 0], [0, 100, 0, 0],
                     [0, 0, 100, 0], [0, 0, 0, 100]]
        expected = {name: dict(valid_miou=.6, confusion=confusion)
                    for name in names}
        actual = {name: dict(value) for name, value in expected.items()}
        self.assertTrue(evaluate.compare_val_replay(actual, expected)['passed'])
        changed = [row[:] for row in confusion]
        changed[2][2] -= 10
        changed[2][1] += 10
        actual['posterior_quality_gate'] = dict(
            valid_miou=.6, confusion=changed)
        gate = evaluate.compare_val_replay(actual, expected)
        self.assertFalse(gate['passed'])
        self.assertFalse(gate['checks']['posterior_quality_gate']['passed'])


if __name__ == '__main__':
    unittest.main()
