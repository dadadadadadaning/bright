"""CPU contracts for selective retrieval and posterior-gated Memory."""
import copy
import inspect
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from . import common
from .memory import (
    POSTERIOR_MODES,
    RETRIEVAL_MODES,
    SelectivePosteriorHead,
    assert_no_query_target_interfaces,
    build_quality_view,
    classify_posterior_regions,
    device_bank,
    make_posterior_heads,
    make_retrieval_heads,
    retrieve,
)
from bda_benchmark.experiments.damageformer_partition_learning_v1 import reproduction as repro


def _token_with_cosine(value, dim=8):
    result = torch.zeros(dim)
    result[0] = value
    result[1] = math.sqrt(max(0., 1. - value * value))
    return result


def tiny_bank(dim=8):
    cosine = (.99, .98, .97, .96, .95, .94, .93, .92)
    classes = torch.tensor([0, 0, 1, 1, 2, 3, 3, 2])
    distribution = F.one_hot(classes, 4).float()
    bank = dict(
        format='ROI_MEMORY_QUALITY_BANK_V2', policy='hard', split='train',
        test_accessed=False, smoke=False,
        identity={'dim': dim, 'space': 'synthetic'},
        image_names=[f'image_{index}' for index in range(8)],
        event_names=['event_a', 'event_b'],
        image_event_ids=[0, 0, 1, 1, 1, 1, 1, 1], per_bucket=4,
        tokens=torch.stack([_token_with_cosine(value, dim)
                            for value in cosine]),
        distribution=distribution,
        labelled_mass=torch.arange(1, 9).float(),
        hard_count=torch.arange(1, 9).long(),
        purity=torch.ones(8), entropy=torch.zeros(8),
        images=torch.arange(8),
        events=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
        regions=torch.arange(8))
    return device_bank(bank, 'cpu')


def quality_bank(dim=8):
    rows, classes, hard = [], [], []
    for class_id in range(4):
        for index, peak in enumerate((.60, .80, .95)):
            row = torch.full((4,), (1. - peak) / 3.)
            row[class_id] = peak
            rows.append(row); classes.append(class_id); hard.append((1, 2, 4)[index])
    distribution = torch.stack(rows)
    purity = distribution.max(-1).values
    entropy = -(distribution * distribution.log()).sum(-1)
    count = len(rows)
    bank = dict(
        policy='hard', split='train', tokens=torch.randn(count, dim),
        distribution=distribution, labelled_mass=torch.linspace(1., 3., count),
        hard_count=torch.tensor(hard), purity=purity, entropy=entropy,
        images=torch.arange(count), events=torch.zeros(count, dtype=torch.long),
        regions=torch.arange(count), dominant_class=torch.tensor(classes))
    return bank


class SelectiveCoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def test_quality_view_is_train_classwise_inclusive_and_nonmutating(self):
        bank = quality_bank()
        all_rows = build_quality_view(bank, quantile=0.)
        self.assertTrue(all_rows['quality_eligible'].all())
        self.assertNotIn('quality_eligible', bank)
        selected = build_quality_view(bank, quantile=.25)
        self.assertEqual(len(selected['quality_thresholds']), 4)
        for row in selected['quality_thresholds']:
            self.assertEqual(row['count'], 3)
            self.assertEqual(row['retained'], 2)
            self.assertIsNotNone(row['purity_min'])
            self.assertIsNotNone(row['entropy_max'])
            self.assertIsNotNone(row['hard_count_min'])
        self.assertEqual(int(selected['quality_eligible'].sum()), 8)
        invalid = dict(bank, split='val')
        with self.assertRaises(ValueError):
            build_quality_view(invalid)

    def test_adaptive_cap_keeps_global_order_without_quota_or_backfill(self):
        bank = tiny_bank()
        # Top quality candidates are 0(c0),1(c0),3(c1),4(c2).  Candidate
        # 5(c3) exists below the global top-4 boundary.  With cap=1, row 1 is
        # rejected, but row 5 must not be pulled in as a backfill.
        bank['quality_eligible'] = torch.tensor(
            [True, True, False, True, True, True, True, True])
        query = torch.tensor([[1.] + [0.] * 7])
        _, _, valid, metadata = retrieve(
            query, bank, -1, -1, policy='adaptive_cap', topk=4,
            preselect=4, per_class_cap=1)
        chosen = metadata['indices'][0, valid[0]].tolist()
        self.assertEqual(chosen, [0, 3, 4])
        self.assertEqual(metadata['dominant_class'][0, valid[0]].tolist(),
                         [0, 1, 2])
        self.assertEqual(int(valid.sum()), 3)
        self.assertFalse((metadata['indices'][valid] == 5).any())
        selected_cosine = metadata['cosine'][valid]
        self.assertTrue(bool((selected_cosine[:-1] >= selected_cosine[1:]).all()))

    def test_global_order_and_source_exclusions(self):
        bank = tiny_bank()
        query = torch.tensor([[1.] + [0.] * 7])
        _, _, valid, metadata = retrieve(
            query, bank, -1, -1, policy='global_hard', topk=3)
        self.assertEqual(metadata['indices'][valid].tolist(), [0, 1, 2])
        _, _, valid, metadata = retrieve(
            query, bank, event_id=0, image_id=0,
            policy='global_hard', topk=8, exclude_same_event=True)
        self.assertTrue((metadata['events'][valid] == 1).all())
        self.assertFalse((metadata['images'][valid] == 0).any())
        _, _, valid, metadata = retrieve(
            query, bank, event_id=0, image_id=0,
            policy='global_hard', topk=8, exclude_same_event=False)
        self.assertFalse((metadata['images'][valid] == 0).any())
        self.assertTrue((metadata['images'][valid] == 1).any())

    def test_random_is_not_conditioned_on_nondefault_cosine_floor(self):
        bank = tiny_bank()
        bank['quality_eligible'] = torch.ones(8, dtype=torch.bool)
        with self.assertRaises(ValueError):
            retrieve(torch.randn(2, 8), bank, -1, -1,
                     policy='adaptive_cap', intervention='random',
                     min_cosine=0.)

    def test_matched_heads_are_independent_identical_and_step0_identity(self):
        query, base = torch.randn(5, 8), torch.randn(5, 4)
        bank = tiny_bank(); bank['quality_eligible'] = torch.ones(8, dtype=torch.bool)
        support, labels, valid, metadata = retrieve(
            query, bank, -1, -1, policy='adaptive_cap')
        retrieval_heads = make_retrieval_heads(8, 'cpu')
        posterior_heads = make_posterior_heads(8, 'cpu')
        for heads in (retrieval_heads, posterior_heads):
            states = [tuple(head.state_dict().values()) for head in heads.values()]
            for state in states[1:]:
                for first, other in zip(states[0], state):
                    torch.testing.assert_close(first, other, rtol=0, atol=0)
            pointers = [{parameter.data_ptr() for parameter in head.parameters()}
                        for head in heads.values()]
            for first in range(len(pointers)):
                for second in range(first + 1, len(pointers)):
                    self.assertFalse(pointers[first] & pointers[second])
        for name, head in retrieval_heads.items():
            logits = (head(query, base, mode='nomemory')
                      if name == 'nomemory' else
                      head(query, base, support, labels, valid, mode='memory'))
            torch.testing.assert_close(logits, base, rtol=0, atol=0)
        for name, head in posterior_heads.items():
            logits = (head(query, base, mode=name)
                      if name == 'query_only' else
                      head(query, base, labels, valid, metadata['cosine'],
                           metadata['purity'], metadata['entropy'], mode=name))
            torch.testing.assert_close(logits, base, rtol=0, atol=0)
        self.assertEqual(tuple(retrieval_heads), RETRIEVAL_MODES)
        self.assertEqual(tuple(posterior_heads), POSTERIOR_MODES)

    def test_all_invalid_falls_back_to_query_only_after_training(self):
        head = SelectivePosteriorHead(8)
        with torch.no_grad():
            head.query_residual[-1].bias.copy_(torch.tensor([.2, -.1, .3, -.2]))
            head.strength.fill_(.8)
        query, base = torch.randn(3, 8), torch.randn(3, 4)
        query_only = head(query, base, mode='query_only')
        valid = torch.zeros(3, 4, dtype=torch.bool)
        invalid = torch.full((3, 4), float('nan'))
        labels = torch.full((3, 4, 4), float('nan'))
        fallback = head(query, base, labels, valid, invalid, invalid, invalid,
                        mode='posterior_quality_gate')
        self.assertTrue(torch.isfinite(fallback).all())
        torch.testing.assert_close(fallback, query_only, rtol=0, atol=0)

    def test_invalid_padding_is_finite_and_support_labels_change_posterior(self):
        head = SelectivePosteriorHead(8)
        with torch.no_grad():
            head.strength.fill_(1.)
        query, base = torch.randn(2, 8), torch.zeros(2, 4)
        valid = torch.tensor([[True, False, False], [True, False, False]])
        cosine = torch.tensor([[.8, float('nan'), float('nan')],
                               [.7, float('nan'), float('nan')]])
        purity = torch.tensor([[1., float('nan'), float('nan')],
                               [1., float('nan'), float('nan')]])
        entropy = torch.tensor([[0., float('nan'), float('nan')],
                                [0., float('nan'), float('nan')]])
        labels_a = torch.full((2, 3, 4), float('nan'))
        labels_b = torch.full((2, 3, 4), float('nan'))
        labels_a[:, 0] = F.one_hot(torch.tensor([0, 0]), 4).float()
        labels_b[:, 0] = F.one_hot(torch.tensor([2, 2]), 4).float()
        output_a = head(query, base, labels_a, valid, cosine, purity, entropy,
                        mode='posterior_nogate')
        output_b = head(query, base, labels_b, valid, cosine, purity, entropy,
                        mode='posterior_nogate')
        self.assertTrue(torch.isfinite(output_a).all())
        self.assertTrue(torch.isfinite(output_b).all())
        self.assertFalse(torch.equal(output_a, output_b))
        self.assertTrue((output_a.argmax(-1) == 0).all())
        self.assertTrue((output_b.argmax(-1) == 2).all())

    def test_shuffle_changes_only_labels_and_is_query_chunk_invariant(self):
        bank = tiny_bank(); bank['quality_eligible'] = torch.ones(8, dtype=torch.bool)
        query = torch.randn(5, 8)
        normal = retrieve(query, bank, -1, -1, policy='adaptive_cap')
        shuffled = None
        seed = None
        for candidate_seed in range(20):
            candidate = retrieve(
                query, bank, -1, -1, policy='adaptive_cap',
                intervention='shuffle_labels',
                generator=torch.Generator().manual_seed(candidate_seed))
            if not torch.equal(candidate[1], normal[1]):
                shuffled, seed = candidate, candidate_seed
                break
        self.assertIsNotNone(shuffled)
        torch.testing.assert_close(shuffled[3]['indices'], normal[3]['indices'])
        torch.testing.assert_close(shuffled[3]['cosine'], normal[3]['cosine'])
        torch.testing.assert_close(shuffled[2], normal[2])

        head = SelectivePosteriorHead(8)
        with torch.no_grad():
            head.strength.fill_(1.)
        tokens = query[None]
        mass = torch.ones(1, len(query))
        base = torch.zeros(1, len(query), 4)
        outputs = []
        for chunk in (1, 32):
            logits, _ = classify_posterior_regions(
                head, tokens, mass, base, bank, ['query_image'],
                ['query_event'], mode='posterior_nogate',
                retrieval_policy='adaptive_cap', topk=8,
                query_chunk=chunk, intervention='shuffle_labels',
                generator=torch.Generator().manual_seed(seed))
            outputs.append(logits)
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)

    def test_query_targets_are_absent_from_every_public_interface(self):
        assert_no_query_target_interfaces()
        forbidden = {'gt', 'target', 'query_gt', 'query_label', 'query_labels'}
        for function in (retrieve, classify_posterior_regions,
                         SelectivePosteriorHead.forward,
                         SelectivePosteriorHead.gate_features):
            self.assertFalse(
                forbidden & set(inspect.signature(function).parameters))

    def test_selective_gate_only_allows_bounded_legacy_region_drift(self):
        def metric():
            return dict(
                valid_miou=.5,
                iou=[.5] * 4,
                precision=[.5] * 4,
                recall=[.5] * 4)

        expected = dict(
            old_pixel=metric(), old_region=metric(), hard=metric())
        actual = dict(
            old_pixel=copy.deepcopy(expected['old_pixel']),
            old_region=copy.deepcopy(expected['old_region']),
            source_hard=copy.deepcopy(expected['hard']))
        actual['old_region']['iou'][3] += 2e-5
        actual['old_region']['precision'][3] += 3e-5
        policy = repro.build_policy(expected)
        bundle = {'summary': {'final_metrics': expected}}
        args = SimpleNamespace(smoke=False)

        with TemporaryDirectory() as directory, mock.patch.object(
                common, 'assert_source', return_value=None):
            first = Path(directory) / 'bounded'
            first.mkdir()
            report = common.selective_initial_gate(
                bundle, {'metrics': actual}, args, first, policy)
            self.assertTrue(report['passed'])
            self.assertFalse(report['all_metric_checks_passed'])
            self.assertEqual(
                set(report['nonblocking_failures']),
                {'old_region/iou/3', 'old_region/precision/3'})
            self.assertEqual(
                report['status'], 'passed_with_bounded_old_region_drift')

            # The exception is deliberately narrow.  The same numerical
            # delta must remain blocking for the task-relevant baselines.
            for baseline in ('old_pixel', 'source_hard'):
                destination = Path(directory) / ('failure_' + baseline)
                destination.mkdir()
                bad = copy.deepcopy(actual)
                bad[baseline]['iou'][3] += 2e-5
                with self.assertRaisesRegex(
                        ValueError, 'Blocking baseline reproduction failed'):
                    common.selective_initial_gate(
                        bundle, {'metrics': bad}, args, destination, policy)

            # A legacy-region deviation beyond the predeclared global cap is
            # still a hard failure, even when the only reason is absolute
            # difference.
            beyond = Path(directory) / 'beyond_cap'
            beyond.mkdir()
            bad = copy.deepcopy(actual)
            bad['old_region']['recall'][3] += repro.MAX_TOL * 1.01
            with self.assertRaisesRegex(
                    ValueError, 'Blocking baseline reproduction failed'):
                common.selective_initial_gate(
                    bundle, {'metrics': bad}, args, beyond, policy)

            second = Path(directory) / 'source_failure'
            second.mkdir()
            bad = copy.deepcopy(actual)
            bad['source_hard']['iou'][3] += 2e-5
            with self.assertRaisesRegex(
                    ValueError, 'Blocking baseline reproduction failed'):
                common.selective_initial_gate(
                    bundle, {'metrics': bad}, args, second, policy)
            failed = __import__('json').loads(
                (second / 'summary.json').read_text())
            self.assertEqual(failed['optimizer_steps'], 0)

    def test_bounded_legacy_region_helper_rejects_anomaly_or_malformed_rows(self):
        valid = dict(
            baseline='old_region', delta=2e-5,
            reason='absolute difference exceeds tolerance',
            history={'history_anomaly': False})
        self.assertTrue(common._bounded_old_region_drift(valid))

        for change in (
                {'baseline': 'old_pixel'},
                {'delta': repro.MAX_TOL * 1.01},
                {'delta': float('nan')},
                {'delta': 10 ** 10000},
                {'reason': ('absolute difference exceeds tolerance; '
                            'metric outside [0, 1]')},
                {'history': {'history_anomaly': True}},
                {'history': {}},
                {'history': None}):
            row = copy.deepcopy(valid)
            row.update(change)
            self.assertFalse(common._bounded_old_region_drift(row), row)
        self.assertFalse(common._bounded_old_region_drift(None))


if __name__ == '__main__':
    unittest.main()

