"""Synthetic CPU end-to-end: dual bank, support audit and 20-step training."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from . import audit, build_bank, common, report, train
from .memory import HEAD_FORMAT, read_banks
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import route_from_q


class SyntheticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bank_audit_and_residual_training(self):
        torch.manual_seed(13)
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        front = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        source = Readout(dim=8).eval().requires_grad_(False)
        bundle = dict(
            front=front, heads={'hard': source}, train=train_names, val=val_names,
            device=torch.device('cpu'),
            manifest={'arguments': {'epsilon': .05, 'threshold': .5},
                      'dim': 8, 'stride': 8},
            summary={'final_metrics': {}, 'final_step': 5000}, provenance={})
        identity = dict(dim=8, front='synthetic_frozen', embedding='synthetic_hard',
                        pool_mode='hard', read_mode='hard', followup_hash='synthetic')
        initial_states = dict(front=copy.deepcopy(front.state_dict()),
                              source=copy.deepcopy(source.state_dict()))

        def loader(dataset, names, train=False, batch=1, workers=0, seed=0):
            batches = []
            for start in range(0, len(names), batch):
                current = list(names[start:start + batch])
                if train and len(current) < batch:
                    continue
                pre = torch.stack([torch.full((3, 4, 4),
                    .1 + .2 * int('event_b' in name)) for name in current])
                post = pre + torch.linspace(0, .2, 16).reshape(1, 4, 4)
                gt = torch.stack([((torch.arange(16).reshape(4, 4)
                    + int('event_b' in name)) % 4) for name in current]).long()
                batches.append((pre, post, torch.zeros_like(gt), gt, current))
            return batches

        def capture(model, pre, post):
            batch = len(pre)
            feature = (torch.arange(256 * 4, dtype=torch.float32)
                       .reshape(1, 256, 2, 2) / 300)
            feature = feature.expand(batch, -1, -1, -1).clone()
            feature += pre.mean((1, 2, 3))[:, None, None, None]
            q = torch.zeros(batch, 9, 4)
            q[:, 0] = .7; q[:, 1] = .3
            ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
            building = torch.tensor([[[[.2, .8], [.8, .2]]]]).expand(batch, -1, -1, -1)
            route = route_from_q(q, ids, 2, (2, 2), building, epsilon=0.)
            region = source(feature, route, (4, 4))
            pixel = torch.softmax(torch.randn(batch, 4, 4, 4,
                generator=torch.Generator().manual_seed(77)), 1)
            return dict(features={'roi': feature}, q=q, ids=ids,
                        tokens=torch.zeros(batch, 2, 8), shape=(2, 2),
                        building=building, pixel=pixel, prob=region)

        def routes(reference, manifest, pool_mode, read_mode):
            route = route_from_q(reference['q'], reference['ids'], 2,
                               reference['shape'], reference['building'], epsilon=0.)
            return route, route

        def assert_frozen(received, expected_hash=None):
            self.assertIs(received, bundle)
            for model, original in ((front, initial_states['front']),
                                    (source, initial_states['source'])):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

        with tempfile.TemporaryDirectory(prefix='quality_memory_v2_') as temporary, ExitStack() as stack:
            root = Path(temporary)
            bank_root, audit_root, train_root = root / 'bank', root / 'audit', root / 'train'
            replacements = (
                ('load_source', mock.Mock(return_value=bundle)),
                ('source_identity', mock.Mock(return_value=identity)),
                ('experiment_hash', mock.Mock(return_value='synthetic_v2_hash')),
                ('assert_v2_immutable', assert_frozen),
                ('loader', loader), ('capture', capture), ('routes', routes),
                ('parse_batch', lambda batch, device: batch),
                ('seed_everything', lambda seed: torch.manual_seed(seed)),
            )
            for name, replacement in replacements:
                stack.enter_context(mock.patch.object(common, name, replacement))
            stack.enter_context(mock.patch.object(
                train, 'baseline_policy', mock.Mock(return_value=None)))
            stack.enter_context(mock.patch.object(
                train, 'initial_gate', mock.Mock(return_value={'passed': None})))
            shared = ['--checkpoint', 'synthetic_front', '--run_root', 'synthetic_source',
                      '--dataset', 'synthetic', '--train_list', 'train.txt',
                      '--val_list', 'val.txt', '--device', 'cpu', '--workers', '0', '--smoke']
            with mock.patch.object(sys, 'argv', ['build_bank', *shared,
                    '--output', str(bank_root), '--per_bucket', '2']), redirect_stdout(io.StringIO()):
                build_bank.main()
            bank_manifest = json.loads((bank_root / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(bank_manifest['complete'])
            self.assertLessEqual(bank_manifest['all_regions'], 2 * 4 * 2)
            self.assertLessEqual(bank_manifest['hard_regions'], 2 * 4 * 2)
            banks, _ = read_banks(bank_root, identity, True, common.digest, train_names)
            self.assertTrue((banks['hard']['hard_count'] > 0).all())
            with mock.patch.object(sys, 'argv', ['audit', *shared,
                    '--output', str(audit_root), '--bank', str(bank_root),
                    '--topk', '8']), redirect_stdout(io.StringIO()):
                audit.main()
            audit_summary = json.loads((audit_root / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(audit_summary['complete'])
            self.assertEqual(audit_summary['image_count'], 2)
            self.assertEqual(set(audit_summary['cases']), {
                'global_all', 'global_hard', 'balanced_hard',
                'balanced_hard_random', 'balanced_hard_shuffle_labels'})
            for row in audit_summary['cases'].values():
                self.assertIn('knn_hard_query_coverage', row)
                self.assertLessEqual(row['knn_supported_queries'],
                                     row['query_labelled_hard'])
            with mock.patch.object(sys, 'argv', ['train', *shared,
                    '--output', str(train_root), '--bank', str(bank_root),
                    '--steps', '20', '--eval_every', '20', '--batch', '2',
                    '--accum', '8', '--query_chunk', '2']), redirect_stdout(io.StringIO()):
                train.main()
            summary = json.loads((train_root / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['final_step'], 20)
            self.assertFalse(summary['test_accessed'])
            expected = {'old_pixel', 'old_region', 'source_hard', *MODES,
                        'balanced_hard_random', 'balanced_hard_shuffle_labels',
                        'balanced_hard_off'}
            self.assertEqual(set(summary['final_metrics']), expected)
            self.assertEqual(summary['support']['balanced_hard_off']['support_coverage'], 0.)
            initial = train.evaluate(bundle, make_heads_for_test(banks),
                {k: device_bank_for_test(v) for k, v in banks.items()},
                loader('synthetic', val_names), smoke_args(), initial=True)
            self.assertTrue(initial['zero_residual_identity_checked'])
            for mode in MODES:
                for suffix in ('best', 'last'):
                    checkpoint = torch.load(train_root / f'{mode}_{suffix}.pth',
                                            map_location='cpu', weights_only=False)
                    self.assertEqual(checkpoint['format'], HEAD_FORMAT)
                    self.assertNotIn('front', checkpoint)
                    self.assertNotIn('optimizer', checkpoint)
                    self.assertNotIn('bank', checkpoint)
            self.assertIn('QUALITY global_hard-global_all', report.render(train_root))
            self.assertIn('Gate:', report.render(audit_root))
            assert_frozen(bundle)


def make_heads_for_test(banks):
    from .memory import make_heads
    return make_heads(banks['all']['tokens'].shape[1], 'cpu')


def device_bank_for_test(bank):
    from .memory import device_bank
    return device_bank(bank, 'cpu')


def smoke_args():
    return type('Args', (), dict(source_head='hard', pool_mode='hard', read_mode='hard',
        topk=8, query_chunk=2, allow_same_event=False, seed=3407, smoke=True))()


# Import after helper definitions so the expected set expression remains clear.
from .memory import MODES


if __name__ == '__main__':
    unittest.main()
