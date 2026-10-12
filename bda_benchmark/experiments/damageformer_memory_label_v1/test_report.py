"""Synthetic reporting checks; these fixtures are not real-data efficacy evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import (
    accumulate, finalize_bucket, new_bucket,
)
from .report import FORMAT, assess, render


def fixture(mode='train', errors=None):
    """Use the real diagnostic bucket schema and exact paired transitions."""
    truth = torch.arange(4).repeat_interleave(20)
    errors = errors or {}
    defaults = dict(old_pixel=8, old_region=8, source_hard=8, knn_only=4,
                    nomemory=6, memory=3, memory_random=10,
                    memory_shuffle_labels=10, memory_off=8,
                    knn_mix=3, knn_random=10, knn_shuffle_labels=10, knn_off=8)
    names = (['old_pixel', 'old_region', 'source_hard', 'knn_only', 'nomemory',
              'memory', 'memory_random', 'memory_shuffle_labels', 'memory_off'] if mode == 'train'
             else ['old_pixel', 'old_region', 'source_hard', 'knn_only', 'knn_mix',
                   'knn_random', 'knn_shuffle_labels', 'knn_off'])
    predictions = {}
    for name in names:
        pred = truth.clone()
        for c in range(4):
            count = errors.get(name, defaults[name])
            count = count[c] if isinstance(count, (list, tuple)) else count
            pred[c * 20:c * 20 + count] = (c + 1) % 4
        predictions[name] = pred
    candidate = 'memory' if mode == 'train' else 'knn_mix'
    prefix = 'memory' if mode == 'train' else 'knn'
    pairs = [('source_hard', name) for name in names if name != 'source_hard']
    pairs += [('old_pixel', name) for name in names if name not in ('old_pixel', 'old_region')]
    pairs += [(candidate, prefix + '_' + kind) for kind in ('random', 'shuffle_labels', 'off')]
    if mode == 'train':
        pairs += [('nomemory', 'memory')]
    bucket = new_bucket(names, pairs)
    accumulate(bucket, truth, predictions, torch.ones_like(truth, dtype=torch.bool), pairs)
    finalized = finalize_bucket(bucket)
    audit = dict(image_count=450, groups={'all': finalized}, per_event={'synthetic_event': finalized},
                 per_image=[dict(image='synthetic_event_0', event='synthetic_event', **finalized)],
                 test_accessed=False)
    summary = dict(format=FORMAT, complete=True, smoke=False, mode=mode, final_step=100,
                   image_count=450, baseline_reproduction_pass=True,
                   frozen_source_verified=True, test_accessed=False,
                   final_metrics=copy.deepcopy(finalized['metrics']))
    return summary, audit


class ReportTests(unittest.TestCase):
    def test_complete_synthetic_evidence_passes_every_separate_gate(self):
        for mode in ('audit', 'train'):
            with self.subTest(mode=mode):
                evidence = assess(*fixture(mode))
                self.assertTrue(evidence['ready'])
                self.assertTrue(evidence['positive_signal'])
                self.assertTrue(all(evidence['gates'].values()))
                self.assertIn('not statistical proof', evidence['conclusion'])
                self.assertEqual(evidence['candidate'], 'memory' if mode == 'train' else 'knn_mix')

    def test_background_only_rescue_is_not_a_positive_signal(self):
        summary, audit = fixture(errors={'nomemory': [8, 6, 6, 6], 'memory': [0, 6, 6, 6]})
        evidence = assess(summary, audit)
        comparison = evidence['comparisons']['nomemory__memory']
        self.assertGreater(comparison['delta']['foreground_miou_pp'], 0)
        self.assertGreater(comparison['rescue_harm']['all']['net'], 0)
        self.assertEqual(comparison['rescue_harm']['gt_foreground']['net'], 0)
        self.assertFalse(evidence['gates']['nomemory:gt_foreground_rescue_exceeds_harm'])
        self.assertFalse(evidence['positive_signal'])

    def test_train_requires_beating_nomemory_not_only_source_hard(self):
        evidence = assess(*fixture(errors={'nomemory': 1, 'memory': 3}))
        self.assertTrue(evidence['gates']['source_hard:foreground_miou_improved'])
        self.assertFalse(evidence['gates']['nomemory:foreground_miou_improved'])
        self.assertEqual(evidence['primary_baseline'], 'nomemory')
        self.assertFalse(evidence['positive_signal'])

    def test_train_also_requires_beating_source_hard(self):
        evidence = assess(*fixture(errors={'source_hard': 1, 'nomemory': 6, 'memory': 3}))
        self.assertTrue(evidence['gates']['nomemory:foreground_miou_improved'])
        self.assertFalse(evidence['gates']['source_hard:foreground_miou_improved'])
        self.assertFalse(evidence['positive_signal'])

    def test_class_harm_stays_signed_and_uses_true_class_rows(self):
        evidence = assess(*fixture(errors={'memory': [1, 1, 9, 1]}))
        row = evidence['comparisons']['nomemory__memory']
        self.assertEqual(row['rescue_harm']['gt_damaged']['rescued'], 0)
        self.assertEqual(row['rescue_harm']['gt_damaged']['harmed'], 3)
        self.assertEqual(row['rescue_harm']['gt_damaged']['net'], -3)
        self.assertEqual(row['rescue_harm']['by_class']['Damaged']['net'], -3)
        self.assertEqual(row['delta']['classes']['Damaged']['tp_change'], -3)
        self.assertEqual(row['delta']['classes']['Damaged']['fn_change'], 3)
        self.assertFalse(evidence['gates']['nomemory:damaged_net_nonnegative'])
        self.assertFalse(evidence['positive_signal'])

    def test_smoke_incomplete_or_partial_val_never_supports_efficacy(self):
        for kind in ('smoke', 'incomplete', 'partial', 'unknown_smoke'):
            with self.subTest(kind=kind):
                summary, audit = fixture()
                if kind == 'smoke':
                    summary['smoke'] = True
                elif kind == 'incomplete':
                    summary['complete'] = False
                elif kind == 'partial':
                    audit['image_count'] = 449
                else:
                    del summary['smoke']
                evidence = assess(summary, audit)
                self.assertFalse(evidence['complete_non_smoke_val450'])
                self.assertFalse(evidence['positive_signal'])

    def test_missing_audit_raises_and_render_reports_not_ready(self):
        summary, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Final audit unavailable'):
            assess(summary, None)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            text = render(directory)
            self.assertIn('Report not ready', text)
            self.assertIn('Positive signal: False', text)
            self.assertIn('missing evidence is not zero', text)

    def test_formal_signal_requires_explicit_source_verification_and_no_test(self):
        for field, value in (('baseline_reproduction_pass', False),
                             ('baseline_reproduction_pass', None),
                             ('frozen_source_verified', False),
                             ('frozen_source_verified', None),
                             ('test_accessed', True), ('test_accessed', None),
                             ('image_count', 449), ('image_count', None)):
            with self.subTest(field=field, value=value):
                summary, audit = fixture()
                summary[field] = value
                self.assertFalse(assess(summary, audit)['positive_signal'])

    def test_early_gate_failure_summary_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, audit = fixture()
            (root / 'summary.json').write_text(json.dumps({'complete': False, 'error': 'baseline mismatch'}), encoding='utf-8')
            (root / 'audit_final.json').write_text(json.dumps(audit), encoding='utf-8')
            text = render(root)
            self.assertIn('Report not ready', text)
            self.assertIn('Positive signal: False', text)

    def test_missing_paired_audit_cannot_be_replaced_with_aggregate_metrics(self):
        summary, audit = fixture()
        del audit['groups']['all']['comparisons']['nomemory__memory']
        evidence = assess(summary, audit)
        self.assertFalse(evidence['ready'])
        self.assertFalse(evidence['positive_signal'])
        self.assertIn('comparison:nomemory__memory', evidence['missing_evidence'])

    def test_missing_source_knn_diagnostic_pair_is_not_silently_omitted(self):
        summary, audit = fixture()
        del audit['groups']['all']['comparisons']['source_hard__knn_only']
        evidence = assess(summary, audit)
        self.assertFalse(evidence['ready'])
        self.assertIn('comparison:source_hard__knn_only', evidence['missing_evidence'])

    def test_undefined_or_missing_damaged_f1_is_preserved_and_rejects_gate(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                summary, audit = fixture()
                metric = audit['groups']['all']['metrics']['memory']
                if missing:
                    del metric['f1']
                else:
                    metric['f1'][2] = None
                evidence = assess(summary, audit)
                self.assertIsNone(evidence['comparisons']['nomemory__memory']['delta']['classes']['Damaged']['f1_pp'])
                self.assertFalse(evidence['gates']['nomemory:damaged_f1_available'])
                self.assertFalse(evidence['positive_signal'])
                json.dumps(evidence, allow_nan=False)

    def test_label_dependence_requires_strict_gain_over_both_interventions(self):
        for name in ('memory_random', 'memory_shuffle_labels'):
            with self.subTest(name=name):
                evidence = assess(*fixture(errors={name: 3}))
                self.assertFalse(evidence['positive_signal'])
                kind = name.removeprefix('memory_')
                self.assertFalse(evidence['gates']['normal_beats_' + kind])
                self.assertEqual(evidence['interventions'][kind]['normal_minus_intervention']['foreground_miou_pp'], 0)

    def test_intervention_deltas_are_normal_minus_control(self):
        evidence = assess(*fixture())
        for kind in ('random', 'shuffle_labels', 'off'):
            delta = evidence['interventions'][kind]['normal_minus_intervention']
            self.assertGreater(delta['foreground_miou_pp'], 0)
            self.assertGreater(delta['accuracy_pp'], 0)

    def test_event_deterioration_reports_count_and_worst_without_threshold(self):
        summary, audit = fixture()
        event = copy.deepcopy(audit['per_event']['synthetic_event'])
        event['metrics']['memory']['valid_miou'] = event['metrics']['nomemory']['valid_miou'] - .025
        audit['per_event']['bad_event'] = event
        evidence = assess(summary, audit)
        result = evidence['comparisons']['nomemory__memory']['per_event']
        self.assertEqual(result['event_count'], 2)
        self.assertEqual(result['deteriorated_count'], 1)
        self.assertEqual(result['worst_event'], 'bad_event')
        self.assertAlmostEqual(result['worst_delta_pp'], -2.5)
        self.assertTrue(evidence['positive_signal'])

    def test_tiny_positive_values_are_compared_before_rounding(self):
        summary, audit = fixture()
        metric = audit['groups']['all']['metrics']['memory']
        base = audit['groups']['all']['metrics']['nomemory']
        metric['valid_miou'] = base['valid_miou'] + 1e-12
        summary['final_metrics']['memory']['valid_miou'] = metric['valid_miou']
        evidence = assess(summary, audit)
        self.assertTrue(evidence['gates']['nomemory:foreground_miou_improved'])
        self.assertGreater(evidence['comparisons']['nomemory__memory']['delta']['foreground_miou_pp'], 0)

    def test_render_includes_all_pairs_controls_and_limits(self):
        summary, audit = fixture()
        summary['controls'] = {'memory': {'alpha': .1, 'temperature': .1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (('summary.json', summary), ('audit_final.json', audit)):
                (root / name).write_text(json.dumps(value), encoding='utf-8')
            (root / 'label_evidence.json').write_text(json.dumps({
                'counts': {'foreground': {'supported_rescue': 12, 'unused_neighbor_opportunity': 4}},
                'candidate': 'memory', 'reference': 'source_hard'}), encoding='utf-8')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            text = render(root)
            self.assertIn('PAIR source_hard__memory', text)
            self.assertIn('PAIR nomemory__memory', text)
            self.assertIn('PAIR old_pixel__memory', text)
            self.assertIn('gt_foreground rescue/harm', text)
            self.assertIn('gt_damaged rescue/harm', text)
            self.assertIn('Controls:', text)
            self.assertIn('not statistical proof', text)
            self.assertIn('Val GT is diagnosis only', text)
            self.assertIn('"supported_rescue": 12', text)
            self.assertIn('observational overlap, not causal proof', text)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_explicit_root_is_required(self):
        result = subprocess.run([sys.executable, '-m',
                                 'bda_benchmark.experiments.damageformer_memory_label_v1.report'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--root', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
