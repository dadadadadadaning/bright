"""Synthetic CPU checks only: isolation, labels, controls and differentiability."""
import copy
import io
import inspect
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import torch
from torch.nn import functional as F

from .memory import (BANK_FORMAT, BalancedReservoir, SupportHead, classify_regions,
                     device_bank, make_heads, retrieve, target_distribution, validate_bank)
from .routing import route_from_q, encode_regions, decode_regions
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective


def tiny_route(epsilon=0.):
    q = torch.full((1, 9, 4), 1. / 9)
    ids = torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone()
    building = torch.tensor([[[[0., 1.], [1., 0.]]]])
    return route_from_q(q, ids, 2, (2, 2), building, epsilon=epsilon)


def tiny_bank(dim=8):
    tokens = torch.arange(4 * dim, dtype=torch.float32).reshape(4, dim) / 20
    return dict(format=BANK_FORMAT, split='train', test_accessed=False, smoke=False,
                identity={'front': 'frozen', 'embedding': 'chosen', 'routing': 'hard'},
                image_names=['train_a', 'train_b', 'train_c', 'train_d'],
                event_names=['event_a', 'event_b'], image_event_ids=[0, 0, 1, 1], per_bucket=2,
                tokens=tokens, distribution=torch.eye(4),
                images=torch.arange(4), events=torch.tensor([0, 0, 1, 1]), regions=torch.arange(4))


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)

    def test_bank_reservoir_is_bounded_balanced_deterministic(self):
        tokens = torch.randn(40, 8)
        labels = F.one_hot(torch.arange(40) % 4, 4).float()
        mass = torch.ones(40)
        reservoirs = [BalancedReservoir(3, seed=7) for _ in range(2)]
        for reservoir in reservoirs:
            reservoir.add(tokens, labels, mass, 0, 0)
            reservoir.add(tokens + 2, labels, mass, 1, 1)
        a, b = [r.finish(['a', 'b'], ['event_a', 'event_b'], {'token': 1}, False) for r in reservoirs]
        self.assertEqual(len(a['tokens']), 2 * 4 * 3)
        for key in ('tokens', 'distribution', 'images', 'events', 'regions'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertTrue(all(row['retained'] == 3 and row['candidates'] == 10 for row in reservoirs[0].counts()))
        self.assertNotIn('q', a)
        self.assertNotIn('features', a)
        validate_bank(a, {'token': 1}, False, ['a', 'b'])

    def test_bank_rejects_legacy_wrong_split_wrong_space_and_smoke(self):
        bank = tiny_bank()
        validate_bank(bank, bank['identity'], False, bank['image_names'])
        cases = [('format', 'SRAC_FOURSTAGE_FRONT_V2'), ('split', 'val'), ('test_accessed', True),
                 ('smoke', True), ('identity', {'different_embedding': 1})]
        for key, value in cases:
            with self.subTest(key=key):
                corrupt = dict(bank, **{key: value})
                with self.assertRaises(ValueError):
                    validate_bank(corrupt, bank['identity'], False, bank['image_names'])
        with self.assertRaises(ValueError):
            validate_bank(bank, bank['identity'], False, ['train_a'])
        with self.assertRaises(ValueError):
            validate_bank(dict(bank, per_bucket=0), bank['identity'], False)
        with self.assertRaises(ValueError):
            validate_bank(dict(bank, events=torch.tensor([1, 0, 1, 1])), bank['identity'], False)
        with self.assertRaises(ValueError):
            validate_bank(dict(bank, identity={'dim': 128}), {'dim': 128}, False)

    def test_dynamic_eighteen_support_targets_ignore_unlabelled_pixels(self):
        route = tiny_route()
        labels = torch.tensor([[[0, 1], [2, 255]]])
        distributions, weights = target_distribution(labels, route)
        self.assertEqual(route['q'].shape, (1, 18, 4))
        expected = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                  [0., 0., 0., 0.], [0., 0., 1., 0.]]])
        torch.testing.assert_close(distributions, expected)
        torch.testing.assert_close(weights, torch.tensor([[1., 1., 0., 1.]]))

    def test_tiny_positive_support_labels_stay_normalized(self):
        route = tiny_route(1e-12)
        # All pixels predict background, so the foreground slots are tiny but
        # legitimate positive supports. They must not acquire sum<1 labels.
        route['q'][:, :9] = 1. / 9 * (1. - 1e-12)
        route['q'][:, 9:] = 1. / 9 * 1e-12
        distributions, weights = target_distribution(torch.tensor([[[0, 1], [2, 3]]]), route)
        self.assertTrue((weights > 0).all())
        torch.testing.assert_close(distributions.sum(-1), torch.ones((1, 4)))
        torch.testing.assert_close(weights[:, 1::2], torch.full((1, 2), 2e-12), atol=1e-18, rtol=1e-5)

    def test_nearest_supports_exclude_same_event_and_always_same_image(self):
        bank = device_bank(tiny_bank(), 'cpu')
        query = bank['tokens'][:1]
        supports, labels, valid = retrieve(query, bank, event_id=0, image_id=0, topk=4)
        selected_classes = labels.argmax(-1)[valid]
        self.assertEqual(set(selected_classes.tolist()), {2, 3})
        self.assertEqual(int(valid.sum()), 2)
        _, labels, valid = retrieve(query, bank, 0, 0, 4, exclude_same_event=False)
        self.assertEqual(set(labels.argmax(-1)[valid].tolist()), {1, 2, 3})
        for intervention in ('random', 'shuffle_labels'):
            support, _, mask = retrieve(query, bank, 0, 0, 4, intervention=intervention,
                                        generator=torch.Generator().manual_seed(9))
            for token in support[mask]:
                self.assertTrue(any(torch.equal(token, allowed) for allowed in bank['tokens'][2:]))

    def test_query_gt_absent_and_retrieval_is_feature_only(self):
        for function in (retrieve, classify_regions, SupportHead.forward):
            parameters = inspect.signature(function).parameters
            self.assertFalse({'gt', 'target', 'query_labels', 'query_gt'} & parameters.keys())
        bank = device_bank(tiny_bank(), 'cpu')
        query = bank['tokens'][1:3]
        first = retrieve(query, bank, -1, -1, 2)
        relabelled = dict(bank, distribution=bank['distribution'].flip(0))
        second = retrieve(query, relabelled, -1, -1, 2)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[2], second[2])
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_empty_support_fallback_finite_and_trainable(self):
        bank = device_bank(tiny_bank(), 'cpu')
        bank['events'] = torch.zeros_like(bank['events'])
        query = torch.randn(3, 8)
        support, distribution, valid = retrieve(query, bank, 0, -1, 4)
        self.assertFalse(valid.any())
        head = SupportHead(8)
        logits = head(query, support, distribution, valid)
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        _, _, disabled = retrieve(query, bank, -1, -1, 4, intervention='off')
        self.assertFalse(disabled.any())

    def test_heads_have_identical_parameters_but_independent_storage(self):
        heads = make_heads(8, 'cpu')
        self.assertEqual(sum(p.numel() for p in heads['memory'].parameters()),
                         sum(p.numel() for p in heads['nomemory'].parameters()))
        before = copy.deepcopy(heads['nomemory'].state_dict())
        for first, second in zip(heads['memory'].parameters(), heads['nomemory'].parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
        with torch.no_grad():
            next(heads['memory'].parameters()).add_(1)
        for name, value in heads['nomemory'].state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_chunked_query_logits_match_unchunked_and_preserve_slot_ids(self):
        bank = device_bank(tiny_bank(), 'cpu')
        head = SupportHead(8).eval()
        tokens = torch.randn(2, 7, 8)
        mass = torch.tensor([[1., 0., 2., 1., 0., 1., 2.], [0., 1., 1., 1., 1., 0., 1.]])
        for mode in ('memory', 'nomemory'):
            a, first = classify_regions(head, tokens, mass, bank, ['query_a', 'query_b'],
                ['event_a', 'event_b'], mode, topk=3, query_chunk=1)
            b, second = classify_regions(head, tokens, mass, bank, ['query_a', 'query_b'],
                ['event_a', 'event_b'], mode, topk=3, query_chunk=100)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            self.assertEqual(first, second)
            self.assertEqual(first['queries'], 10)
            torch.testing.assert_close(a[mass == 0], torch.zeros((4, 4)))
        with self.assertRaises(ValueError):
            classify_regions(head, tokens, mass, bank, ['query_a', 'query_b'],
                             ['event_a', 'event_b'], 'typo')

    def test_head_backward_through_dynamic_unpool_keeps_embedding_frozen(self):
        source = Readout(dim=8).eval().requires_grad_(False)
        before = copy.deepcopy(source.state_dict())
        feature = torch.randn(1, 256, 2, 2, requires_grad=True)
        route = tiny_route(.05)
        with torch.no_grad():
            encoded = encode_regions(source, feature, route)
        heads = make_heads(8, 'cpu')
        bank = device_bank(tiny_bank(), 'cpu')
        for mode, head in heads.items():
            logits, stats = classify_regions(head, encoded['tokens'], encoded['mass'], bank,
                ['query'], ['event_a'], mode, query_chunk=1)
            probabilities = decode_regions(logits, route, encoded['mass'], (4, 4))
            labels = (torch.arange(16).reshape(1, 4, 4) % 4).long()
            loss = objective(probabilities, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in head.parameters()))
            torch.testing.assert_close(probabilities.sum(1), torch.ones((1, 4, 4)))
        self.assertIsNone(feature.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in source.parameters()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_mocked_cpu_smoke_bank_training_and_report_entrypoints(self):
        """Actual bank IO, 20 tiny paired steps, final diagnostics and report.

        Only source loading and image loading are mocked. No user files,
        benchmark imagery, backbone execution or GPU device are involved.
        """
        from . import build_bank, common, train_memory, report
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        front = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        source = Readout(dim=8).eval().requires_grad_(False)
        bundle = dict(front=front, heads={'hard': source}, train=train_names, val=val_names,
                      device=torch.device('cpu'), summary={}, provenance={},
                      manifest={'arguments': {'epsilon': .05, 'threshold': .5}, 'dim': 8})
        identity = {'dim': 8, 'front': 'synthetic_frozen', 'embedding': 'synthetic_fixed',
                    'pool_mode': 'hard', 'read_mode': 'hard', 'followup_hash': 'synthetic_test'}
        original_states = {'front': copy.deepcopy(front.state_dict()), 'head': copy.deepcopy(source.state_dict())}
        loader_calls = []

        def assert_source(received):
            self.assertIs(received, bundle)
            for key, model in (('front', front), ('head', source)):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original_states[key][name], rtol=0, atol=0)

        def loader(dataset, names, train=False, batch=1, workers=0, seed=0):
            loader_calls.append((train, list(names), batch))
            batches = []
            for start in range(0, len(names), batch):
                current = list(names[start:start + batch])
                if train and len(current) < batch:
                    continue
                pre = torch.stack([torch.full((3, 4, 4), .1 if 'event_a' in name else .2) for name in current])
                post = pre + .1
                labels = torch.stack([(torch.arange(16).reshape(4, 4) + int('event_b' in name)) % 4
                                      for name in current]).long()
                batches.append((pre, post, torch.zeros_like(labels), labels, current))
            return batches

        def capture(model, pre, post):
            batch = len(pre)
            feature = torch.arange(256 * 2 * 2, dtype=torch.float32).reshape(1, 256, 2, 2) / 500
            feature = feature.expand(batch, -1, -1, -1) + pre.mean((1, 2, 3))[:, None, None, None]
            return dict(features={'roi': feature}, q=torch.full((batch, 9, 4), 1. / 9),
                        ids=torch.tensor([[0, 0, 1, 1]]).expand(9, -1).clone(),
                        tokens=torch.zeros((batch, 2, 8)), shape=(2, 2),
                        building=torch.tensor([[[[0., 1.], [1., 0.]]]]).expand(batch, -1, -1, -1),
                        pixel=torch.zeros((batch, 4, 4, 4)))

        with tempfile.TemporaryDirectory(prefix='roi_memory_cpu_test_') as temporary, ExitStack() as stack:
            root = Path(temporary)
            bank_root, output = root / 'bank', root / 'paired'
            for name, replacement in (
                ('load_source', mock.Mock(return_value=bundle)),
                ('assert_source', assert_source),
                ('source_identity', mock.Mock(return_value=identity)),
                ('experiment_hash', mock.Mock(return_value='synthetic_test')),
                ('loader', loader), ('capture', capture),
                ('parse_batch', lambda batch, device: batch),
                ('seed_everything', lambda seed: torch.manual_seed(seed)),
            ):
                stack.enter_context(mock.patch.object(common, name, replacement))
            shared = ['--checkpoint', 'synthetic_front', '--run_root', 'synthetic_source',
                      '--dataset', 'synthetic', '--train_list', 'train_set.txt', '--val_list', 'val_set.txt',
                      '--device', 'cpu', '--workers', '0', '--smoke']
            with mock.patch.object(sys, 'argv', ['build_bank', *shared, '--output', str(bank_root), '--per_bucket', '2']), redirect_stdout(io.StringIO()):
                build_bank.main()
            bank_manifest = json.loads((bank_root / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(bank_manifest['complete'])
            self.assertEqual(bank_manifest['image_names'], train_names)
            self.assertEqual(bank_manifest['image_count'], 2)
            self.assertTrue(bank_manifest['smoke'])
            self.assertFalse(any(path.is_dir() for path in bank_root.iterdir()))
            with mock.patch.object(sys, 'argv', ['train_memory', *shared, '--output', str(output),
                        '--bank', str(bank_root), '--steps', '20', '--eval_every', '20',
                        '--batch', '2', '--accum', '8', '--query_chunk', '2']), redirect_stdout(io.StringIO()):
                train_memory.main()
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['smoke'])
            self.assertEqual(summary['final_step'], 20)
            self.assertFalse(summary['test_accessed'])
            self.assertEqual(set(summary['final_metrics']), {'old_pixel', 'original_region', 'frozen_region',
                              'nomemory', 'memory', 'memory_random', 'memory_shuffle_labels', 'memory_off'})
            self.assertEqual(summary['support']['memory_off']['support_coverage'], 0.)
            self.assertGreater(summary['support']['memory']['support_coverage'], 0.)
            for mode in ('memory', 'nomemory'):
                for suffix in ('best', 'last'):
                    checkpoint = torch.load(output / (mode + '_' + suffix + '.pth'), map_location='cpu', weights_only=False)
                    self.assertIn('head', checkpoint)
                    self.assertNotIn('optimizer', checkpoint)
                    self.assertNotIn('front', checkpoint)
                    self.assertNotIn('bank', checkpoint)
                    self.assertTrue(all(not key.startswith('embedding.') for key in checkpoint['head']))
            final_audit = json.loads((output / 'audit_final.json').read_text(encoding='utf-8'))
            self.assertEqual(final_audit['image_count'], 2)
            self.assertIn('nomemory__memory', final_audit['groups']['all']['comparisons'])
            rendered = report.render(output)
            self.assertIn('PRIMARY memory minus nomemory', rendered)
            self.assertIn('smoke is execution evidence', rendered)
            self.assertTrue(all(set(names).issubset(set(train_names if train else train_names + val_names))
                                for train, names, _ in loader_calls))
            assert_source(bundle)


if __name__ == '__main__':
    unittest.main()
