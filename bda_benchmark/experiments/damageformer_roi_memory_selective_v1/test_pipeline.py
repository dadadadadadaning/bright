"""Synthetic CPU audit plus 20-step retrieval/posterior training flows."""
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
from torch.nn import functional as F

from . import audit, common, report, train
from .memory import (
    BANK_FORMAT,
    HEAD_FORMAT,
    POSTERIOR_MODES,
    RETRIEVAL_MODES,
)
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    DualQualityReservoir,
    validate_bank,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import route_from_q


class SyntheticSelectivePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_audit_and_both_training_stages(self):
        torch.manual_seed(53)
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        front = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        source = Readout(dim=8).eval().requires_grad_(False)
        bundle = dict(
            front=front, heads={'hard': source}, train=train_names,
            val=val_names, device=torch.device('cpu'),
            manifest={'arguments': {'epsilon': .05, 'threshold': .5},
                      'dim': 8, 'stride': 8},
            summary={'final_metrics': {}, 'final_step': 5000}, provenance={})
        identity = dict(
            dim=8, front='synthetic_frozen', embedding='synthetic_hard',
            pool_mode='hard', read_mode='hard', followup_hash='synthetic')
        original = dict(front=copy.deepcopy(front.state_dict()),
                        source=copy.deepcopy(source.state_dict()))

        def loader(dataset, names, train=False, batch=1, workers=0, seed=0):
            batches = []
            for begin in range(0, len(names), batch):
                selected = list(names[begin:begin + batch])
                if train and len(selected) < batch:
                    continue
                pre = torch.stack([
                    torch.full((3, 4, 4),
                               .1 + .2 * int('event_b' in name))
                    for name in selected])
                post = pre + torch.linspace(0, .2, 16).reshape(1, 4, 4)
                gt = torch.stack([
                    ((torch.arange(16).reshape(4, 4)
                      + int('event_b' in name)) % 4)
                    for name in selected]).long()
                batches.append((pre, post, torch.zeros_like(gt), gt, selected))
            return batches

        def capture(model, pre, post):
            batch = len(pre)
            feature = (torch.arange(256 * 4, dtype=torch.float32)
                       .reshape(1, 256, 2, 2) / 300.)
            feature = feature.expand(batch, -1, -1, -1).clone()
            feature += pre.mean((1, 2, 3))[:, None, None, None]
            q = torch.zeros(batch, 9, 4)
            q[:, 0] = .7; q[:, 1] = .3
            ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
            building = torch.tensor(
                [[[[.2, .8], [.8, .2]]]]).expand(batch, -1, -1, -1)
            route = route_from_q(q, ids, 2, (2, 2), building, epsilon=0.)
            region = source(feature, route, (4, 4))
            pixel = torch.softmax(torch.randn(
                batch, 4, 4, 4,
                generator=torch.Generator().manual_seed(77)), 1)
            return dict(
                features={'roi': feature}, q=q, ids=ids,
                tokens=torch.zeros(batch, 2, 8), shape=(2, 2),
                building=building, pixel=pixel, prob=region)

        def routes(reference, manifest, pool_mode, read_mode):
            route = route_from_q(
                reference['q'], reference['ids'], 2, reference['shape'],
                reference['building'], epsilon=0.)
            return route, route

        def assert_frozen(received, expected_hash=None,
                          expected_parent_hash=None):
            self.assertIs(received, bundle)
            for model, state in ((front, original['front']),
                                 (source, original['source'])):
                self.assertFalse(model.training)
                self.assertTrue(all(
                    not parameter.requires_grad and parameter.grad is None
                    for parameter in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)

        with tempfile.TemporaryDirectory(
                prefix='selective_memory_v1_') as temporary, ExitStack() as stack:
            root = Path(temporary)
            bank_root = root / 'bank'
            audit_root = root / 'audit'
            retrieval_root = root / 'retrieval'
            posterior_root = root / 'posterior'
            self._write_bank(bank_root, train_names, identity)

            replacements = (
                ('load_source', mock.Mock(return_value=bundle)),
                ('source_identity', mock.Mock(return_value=identity)),
                ('experiment_hash', mock.Mock(return_value='synthetic_selective')),
                ('parent_experiment_hash', mock.Mock(return_value='synthetic_parent')),
                ('assert_selective_immutable', assert_frozen),
                ('loader', loader), ('capture', capture), ('routes', routes),
                ('parse_batch', lambda batch, device: batch),
                ('seed_everything', lambda seed: torch.manual_seed(seed)),
            )
            for name, replacement in replacements:
                stack.enter_context(mock.patch.object(common, name, replacement))
            stack.enter_context(mock.patch.object(
                train, 'baseline_policy', mock.Mock(return_value=None)))
            stack.enter_context(mock.patch.object(
                common, 'selective_initial_gate',
                mock.Mock(return_value={'passed': None})))

            shared = [
                '--checkpoint', 'synthetic_front',
                '--run_root', 'synthetic_source', '--dataset', 'synthetic',
                '--train_list', 'train.txt', '--val_list', 'val.txt',
                '--device', 'cpu', '--workers', '0', '--smoke',
                '--source_head', 'hard', '--pool_mode', 'hard',
                '--read_mode', 'hard', '--bank', str(bank_root),
                '--topk', '8', '--preselect', '8',
                '--per_class_cap', '4', '--quality_quantile', '0.0',
                '--min_cosine', '-1', '--temperature', '0.07',
            ]
            with mock.patch.object(
                    sys, 'argv', ['audit', *shared,
                                  '--output', str(audit_root)]), \
                    redirect_stdout(io.StringIO()):
                audit.main()
            audit_summary = json.loads(
                (audit_root / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(audit_summary['complete'])
            self.assertEqual(audit_summary['image_count'], 2)
            self.assertEqual(set(audit_summary['cases']), {
                'global_hard', 'fixed_balanced', 'quality_adaptive',
                'quality_adaptive_random',
                'quality_adaptive_shuffle_labels', 'quality_adaptive_off'})
            self.assertEqual(
                audit_summary['cases']['quality_adaptive_off']['support_coverage'],
                0.)

            self._run_stage(shared, retrieval_root, 'retrieval')
            self._run_stage(shared, posterior_root, 'posterior')
            retrieval = json.loads(
                (retrieval_root / 'summary.json').read_text(encoding='utf-8'))
            posterior = json.loads(
                (posterior_root / 'summary.json').read_text(encoding='utf-8'))
            self._check_stage(retrieval_root, retrieval, RETRIEVAL_MODES,
                              'adaptive_cap')
            self._check_stage(posterior_root, posterior, POSTERIOR_MODES,
                              'posterior_quality_gate')
            self.assertEqual(
                posterior['support']['query_only']['support_coverage'], 0.)
            self.assertIn('quality_adaptive', report.render(audit_root))
            self.assertIn('Same-step gains versus nomemory',
                          report.render(retrieval_root))
            self.assertIn('Same-step gains versus query_only',
                          report.render(posterior_root))
            self.assertIn('gate status:', report.render(posterior_root))
            assert_frozen(bundle)

    @staticmethod
    def _write_bank(root, image_names, identity):
        root.mkdir(parents=True)
        reservoir = DualQualityReservoir(per_bucket=2, seed=19)
        for image_id, event_id in enumerate((0, 1)):
            generator = torch.Generator().manual_seed(100 + image_id)
            tokens = F.normalize(torch.randn(8, 8, generator=generator), dim=-1)
            distribution = torch.eye(4).repeat(2, 1)
            mass = torch.linspace(1., 2., 8)
            hard = torch.arange(1, 9)
            reservoir.add(
                tokens, distribution, mass, hard,
                distribution.max(-1).values, torch.zeros(8),
                event_id, image_id)
        events = ['event_a', 'event_b']
        banks = {
            policy: reservoir.finish(
                policy, image_names, events, identity, smoke=True)
            for policy in ('all', 'hard')}
        for policy, bank in banks.items():
            validate_bank(bank, identity, True, policy, image_names)
            common.save_pt(root / f'bank_{policy}.pt', bank)
        manifest = dict(
            format=BANK_FORMAT, complete=True, split='train', smoke=True,
            test_accessed=False, identity=identity,
            image_count=2, image_names=list(image_names), event_names=events,
            per_bucket=2, dim=8, bank_limit_per_policy=16,
            all_regions=len(banks['all']['tokens']),
            hard_regions=len(banks['hard']['tokens']))
        manifest['all_sha256'] = common.digest(root / 'bank_all.pt')
        manifest['hard_sha256'] = common.digest(root / 'bank_hard.pt')
        common.save_json(root / 'manifest.json', manifest)

    @staticmethod
    def _run_stage(shared, output, stage):
        arguments = [
            'train', *shared, '--output', str(output), '--stage', stage,
            '--steps', '20', '--eval_every', '20', '--batch', '2',
            '--accum', '8', '--query_chunk', '2',
            '--posterior_retrieval', 'adaptive_cap']
        with mock.patch.object(sys, 'argv', arguments), \
                redirect_stdout(io.StringIO()):
            train.main()

    def _check_stage(self, root, summary, modes, primary):
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['final_step'], 20)
        self.assertFalse(summary['test_accessed'])
        expected = {'old_pixel', 'old_region', 'source_hard', *modes,
                    primary + '_random', primary + '_shuffle_labels',
                    primary + '_off'}
        self.assertEqual(set(summary['final_metrics']), expected)
        self.assertEqual(summary['support'][primary + '_off']['support_coverage'],
                         0.)
        for mode in modes:
            for suffix in ('best', 'last'):
                checkpoint = torch.load(
                    root / f'{mode}_{suffix}.pth', map_location='cpu',
                    weights_only=False)
                self.assertEqual(checkpoint['format'], HEAD_FORMAT)
                self.assertNotIn('front', checkpoint)
                self.assertNotIn('optimizer', checkpoint)
                self.assertNotIn('bank', checkpoint)


if __name__ == '__main__':
    unittest.main()

