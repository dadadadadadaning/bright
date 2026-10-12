"""Tiny CPU execution test of real Memory V2 bank/training/output pipelines.

Only data/front access and external provenance are mocked. Region embedding,
routing, labels, bank validation/serialization, retrieval, residual heads,
loss/backward, diagnostics and generated checkpoints remain real.
"""
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from . import build_bank, common, train
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import read_bank


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_smoke_bank_and_paired_twenty_step_training(self):
        torch.manual_seed(73)
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        front = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        source_head = Readout(dim=8).eval().requires_grad_(False)
        bundle = dict(front=front, heads={'hard': source_head}, train=train_names,
            val=val_names, device=torch.device('cpu'), summary={},
            provenance={'source_head_step': 5000},
            manifest={'arguments': {'epsilon': .05, 'threshold': .5}, 'dim': 8, 'stride': 1})
        identity = dict(dim=8, front='synthetic_frozen', embedding='synthetic_fixed',
            pool_mode='hard', read_mode='hard', followup_hash='synthetic_old_code')
        before = {'front': copy.deepcopy(front.state_dict()),
                  'head': copy.deepcopy(source_head.state_dict())}
        features = torch.randn(1, 256, 2, 2)
        loader_calls = []
        integrity_calls = []

        def assert_source(received):
            self.assertIs(received, bundle)
            integrity_calls.append(True)
            for key, model in (('front', front), ('head', source_head)):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key][name], atol=0, rtol=0)

        def loader(dataset, names, train=False, batch=1, workers=0, seed=0):
            names = list(names)
            self.assertTrue(set(names).issubset(set(train_names if train else train_names + val_names)))
            loader_calls.append((train, names, batch))
            result = []
            for start in range(0, len(names), batch):
                current = names[start:start+batch]
                if train and len(current) < batch:
                    continue
                pre = torch.stack([torch.full((3, 4, 4), .1 if 'event_a' in name else .2)
                                   for name in current])
                labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1],
                                       [2, 2, 3, 3], [2, 2, 3, 3]])
                gt = torch.stack([(labels + int('event_b' in name)) % 4 for name in current])
                result.append((pre, pre+.1, (gt > 0).long(), gt, current))
            return result

        def capture(model, pre, post):
            self.assertIs(model, front)
            batch = len(pre)
            f = features.expand(batch, -1, -1, -1) + pre.mean((1, 2, 3))[:, None, None, None]
            # K=3, but candidates only use regions 0/1. The last two split
            # slots remain inactive and must preserve original base logits.
            reference = dict(features={'roi': f}, q=torch.full((batch, 9, 4), 1./9),
                ids=torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone(),
                tokens=torch.zeros(batch, 3, 8), shape=(2, 2),
                building=torch.tensor([[[[0., 1.], [1., 0.]]]]).expand(batch, -1, -1, -1),
                pixel=torch.zeros(batch, 4, 4, 4))
            route, _ = common.source.routes(reference, bundle['manifest'], 'hard', 'hard')
            reference['prob'] = source_head(f, route, (4, 4))
            return reference

        with tempfile.TemporaryDirectory(prefix='region_memory_v2_cpu_') as temporary, ExitStack() as stack:
            root = Path(temporary)
            bank_root, output = root/'bank', root/'paired'
            for name, replacement in (
                ('load_source', mock.Mock(return_value=bundle)),
                ('assert_source', assert_source),
                ('source_identity', mock.Mock(return_value=identity)),
                ('experiment_hash', mock.Mock(return_value='synthetic_old_code')),
                ('loader', loader), ('capture', capture),
                ('parse_batch', lambda batch, device: batch),
                ('seed_everything', lambda seed: torch.manual_seed(seed)),
            ):
                stack.enter_context(mock.patch.object(common.source, name, replacement))
            for module in (common, build_bank, train):
                stack.enter_context(mock.patch.object(module, 'implementation_hash',
                                                       return_value='synthetic_new_code'))
            shared = dict(checkpoint='synthetic_front', run_root='synthetic_source',
                dataset='synthetic', train_list='train_set.txt', val_list='val_set.txt',
                source_head='hard', pool_mode='hard', read_mode='hard', workers=0,
                device='cpu', smoke=True, seed=3407)
            with redirect_stdout(io.StringIO()):
                build_bank.run(SimpleNamespace(**shared, output=str(bank_root), per_bucket=2))
            # Exercise the old reader/metadata contract on the NEW builder's
            # actual files: no mocked read_bank, serialization, hashes or labels.
            bank, bank_manifest = read_bank(bank_root, identity, True, train_names)
            self.assertTrue(bank_manifest['complete'])
            self.assertEqual(bank_manifest['builder'], 'region_memory_v2')
            self.assertEqual(bank['image_names'], train_names)
            self.assertEqual(bank['event_names'], ['event_a', 'event_b'])
            self.assertEqual(bank_manifest['image_count'], 2)
            self.assertLessEqual(len(bank['tokens']), 2*4*2)
            self.assertFalse(any(p.is_dir() for p in bank_root.iterdir()))
            bank_bytes = (bank_root/'bank.pt').read_bytes()

            args = SimpleNamespace(**shared, output=str(output), bank=str(bank_root),
                steps=20, eval_every=20, batch=2, accum=8, lr=1e-3,
                topk=3, query_chunk=2, allow_same_event=False)
            with redirect_stdout(io.StringIO()):
                train.run(args)
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['phase'], 'complete')
            self.assertEqual(summary['optimizer_steps'], 20)
            self.assertEqual(summary['final_step'], 20)
            self.assertTrue(summary['smoke'])
            self.assertFalse(summary['test_accessed'])
            self.assertTrue(summary['frozen_source_verified'])
            self.assertEqual(set(summary['final_metrics']), {'old_pixel', 'old_region', 'source_hard',
                'nomemory', 'memory', 'memory_random', 'memory_shuffle_labels', 'memory_off'})
            self.assertEqual(summary['support']['memory_off']['support_coverage'], 0.)
            self.assertGreater(summary['support']['memory']['support_coverage'], 0.)
            self.assertTrue(all(summary['training_verification']['weights_changed'].values()))
            self.assertTrue(all(value > 0 for value in
                                summary['training_verification']['nonzero_gradient_steps'].values()))
            initial = json.loads((output/'initial.json').read_text(encoding='utf-8'))
            self.assertTrue(initial['zero_residual_identity_checked'])
            for name in ('memory', 'nomemory'):
                self.assertEqual(initial['metrics'][name], initial['metrics']['source_hard'])
                self.assertEqual(initial['changed_pixel_fraction_vs_source'][name], 0.)
                for suffix in ('best', 'last'):
                    saved = torch.load(output/(name+'_'+suffix+'.pth'), map_location='cpu', weights_only=False)
                    self.assertEqual(saved['step'], 20)
                    self.assertIn('head', saved)
                    for forbidden in ('front', 'optimizer', 'bank', 'embedding', 'source_classifier'):
                        self.assertNotIn(forbidden, saved)
                    self.assertFalse(any(k.startswith(('embedding.', 'encoder_', 'base.'))
                                         for k in saved['head']))
            gate = json.loads((output/'reproduction.json').read_text(encoding='utf-8'))
            self.assertEqual(gate['status'], 'smoke_not_full_val')
            self.assertTrue(gate['source_integrity_pass'])
            audit = json.loads((output/'audit_final.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['image_count'], 2)
            self.assertIn('nomemory__memory', audit['groups']['all']['comparisons'])
            self.assertEqual((bank_root/'bank.pt').read_bytes(), bank_bytes)
            self.assertIn((True, train_names, 2), loader_calls)
            self.assertGreaterEqual(len(integrity_calls), 4)
            assert_source(bundle)


if __name__ == '__main__':
    unittest.main()
