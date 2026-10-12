"""CPU report contracts, including compatibility with real diagnostic output."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .report import GROUPS, PAIRS, _audit_lines, metric_line, partition_quality_lines, read_rows, render, same_step_gains


def fixture():
    metric = dict(valid_miou=.6, iou=[.9, .7, .4, .7],
                  precision=[.9, .8, .5, .8], recall=[.9, .7, .4, .7],
                  f1=[.9, .7467, .4444, .7467])
    values = {name: copy.deepcopy(metric) for name in ('old_pixel', 'old_region', 'hard', 'uniform', 'sar')}
    values['uniform']['valid_miou'] = .61
    values['sar']['valid_miou'] = .63
    manifest = dict(format='PARTITION_LEARNING_V1', suite='support',
                    variants=['hard', 'uniform', 'sar'], arguments=dict(steps=5000, eval_every=500, smoke=False))
    summary = dict(complete=True, suite='support', smoke=False, final_step=5000,
                   best={'hard': .65, 'uniform': .67, 'sar': .68},
                   best_steps={'hard': 1000, 'uniform': 2000, 'sar': 3000},
                   final_metrics=values, same_step_gains_pp={'hard__uniform': 1., 'uniform__sar': 2., 'hard__sar': 3.},
                   weights_changed={'hard': {'readout': True}}, nonzero_gradient_steps={'hard': 5000},
                   immutable_front_verified=True, elapsed_seconds=5., peak_cuda_allocated_bytes=1024,
                   test_accessed=False)
    row = dict(step=5000, image_count=450, metrics=values, routing={'sar': {'empty_fallback': 10}},
               q_drift={'hard': {'max_abs': 0.}})
    return manifest, summary, row


def write_run(root, manifest, summary, rows):
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (root / 'validation.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    (root / 'audit_final.json').write_text(json.dumps(dict(image_count=450, groups={}, per_event={})), encoding='utf-8')


def quality_fixture():
    values = dict(native_valid_area=1000., native_majority_mass=980.,
                  foreground_mass=30., foreground_receives_background_sum=10.,
                  background_receives_foreground_sum=10., hard_used_regions=100,
                  hard_fragmented_regions=20, hard_connected_components=125, image_count=450)
    for c, (mass, same) in enumerate(((970., 960.), (10., 8.), (10., 2.), (10., 6.))):
        values[f'class_{c}_mass'] = mass
        values[f'class_{c}_self_sum'] = same
    return values


class ReportContracts(unittest.TestCase):
    def test_class_tradeoffs_and_missing_are_explicit(self):
        metric = fixture()[2]['metrics']['hard']
        text = metric_line('hard', metric)
        for value in ('Intact=70.0000%', 'Damaged=40.0000%', 'Destroyed=70.0000%',
                      'Damaged P=50.0000%, R=40.0000%, F1=44.4400%'):
            self.assertIn(value, text)
        self.assertIn('Damaged P=N/A, R=N/A, F1=N/A', metric_line('missing', {}))
        self.assertIsNone(same_step_gains({'hard': metric}, ('hard__sar',))['hard__sar'])

    def test_complete_report_uses_same_step_not_independent_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, *fixture()[:2], [fixture()[2]])
            text = render(root)
            self.assertIn('Primary completed formal endpoint available: True', text)
            self.assertIn('Final same-step metrics: step=5000', text)
            self.assertIn('hard__sar=+3.0000', text)
            self.assertIn('Secondary best (separate steps; never the primary comparison)', text)
            self.assertIn('"empty_fallback": 10', text)
            self.assertIn('Q drift:', text)
            self.assertIn('Completed endpoint audit: image_count=450', text)

    def test_incomplete_summary_cannot_promote_existing_audit_or_target_row(self):
        manifest, summary, row = fixture()
        summary['complete'] = False
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [row])
            text = render(root)
            self.assertIn('Primary completed formal endpoint available: False', text)
            self.assertIn('Latest available validation (provisional', text)
            self.assertNotIn('Final same-step metrics:', text)
            self.assertNotIn('Completed endpoint audit:', text)
            self.assertIn('Incomplete run', text)

    def test_formal_endpoint_rejects_missing_or_mismatched_evidence(self):
        changes = ('step', 'metrics', 'missing_branch', 'immutable', 'test', 'format', 'suite',
                   'manifest_suite', 'variants', 'short_val', 'short_protocol', 'unknown_smoke')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                manifest, summary, row = copy.deepcopy(fixture())
                # Remove aliases between row and summary metrics before corruption.
                summary['final_metrics'] = copy.deepcopy(summary['final_metrics'])
                if change == 'step':
                    summary['final_step'] = 4500
                elif change == 'metrics':
                    summary['final_metrics']['sar']['valid_miou'] = .99
                elif change == 'missing_branch':
                    del row['metrics']['uniform']
                elif change == 'immutable':
                    summary['immutable_front_verified'] = False
                elif change == 'test':
                    del summary['test_accessed']
                elif change == 'format':
                    manifest['format'] = 'ROI_EXCHANGE_V1'
                elif change == 'suite':
                    summary['suite'] = 'unknown'
                elif change == 'manifest_suite':
                    manifest['suite'] = 'structure'
                elif change == 'variants':
                    manifest['variants'] = ['hard', 'sar']
                elif change == 'short_val':
                    row['image_count'] = 2
                elif change == 'short_protocol':
                    manifest['arguments']['eval_every'] = 20
                else:
                    del summary['smoke']
                root = Path(temp)
                write_run(root, manifest, summary, [row])
                text = render(root)
                self.assertIn('Primary completed formal endpoint available: False', text)
                self.assertNotIn('Final same-step metrics:', text)

    def test_smoke_even_if_only_recorded_in_manifest_is_not_formal(self):
        manifest, summary, row = fixture()
        manifest['arguments']['smoke'] = True
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [row])
            text = render(root)
            self.assertIn('Primary completed formal endpoint available: False', text)
            self.assertIn('Smoke validates execution only', text)

    def test_broken_tail_does_not_hide_valid_rows_or_invent_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, summary, row = fixture()
            summary['complete'] = False
            write_run(root, manifest, summary, [row])
            with (root / 'validation.jsonl').open('a', encoding='utf-8') as stream:
                stream.write('{"step":\n{"step": 5500, "metrics": {"x": NaN}}\n')
            warnings = []
            self.assertEqual(len(read_rows(root / 'validation.jsonl', warnings)), 1)
            self.assertEqual(len(warnings), 2)
            self.assertIn('skipped incomplete/invalid line', render(root))

    def test_inconsistent_recorded_gain_is_flagged(self):
        manifest, summary, row = fixture()
        summary['same_step_gains_pp']['hard__sar'] = 30.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [row])
            text = render(root)
            self.assertIn('hard__sar=+3.0000', text)
            self.assertIn('hard__sar: summary gain missing/inconsistent', text)

    def test_all_three_suites_select_their_own_pairs(self):
        for suite, names in GROUPS.items():
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temp:
                manifest, summary, row = fixture()
                manifest.update(suite=suite, variants=list(names))
                summary['suite'] = suite
                metric = row['metrics']['hard']
                row['metrics'] = {name: copy.deepcopy(metric) for name in ('old_pixel', 'old_region') + names}
                summary['final_metrics'] = copy.deepcopy(row['metrics'])
                summary['same_step_gains_pp'] = {pair: 0. for pair in PAIRS[suite]}
                root = Path(temp)
                write_run(root, manifest, summary, [row])
                text = render(root)
                self.assertIn('Primary completed formal endpoint available: True', text)
                for pair in PAIRS[suite]:
                    self.assertIn(pair + '=+0.0000', text)

    def test_partition_quality_separates_background_and_all_damage_grades(self):
        text = '\n'.join(partition_quality_lines({'sar': quality_fixture()}, ('sar',)))
        for value in ('Intact: same-class=80.0000%', 'Damaged: same-class=20.0000%',
                      'Destroyed: same-class=60.0000%',
                      'Foreground (all grades): same-grade=53.3333%; receives-background=33.3333%',
                      'Background only: receives-foreground=1.0309%',
                      'Secondary global majority purity (background-weighted): 98.0000%',
                      'fragmented=20.0000%', 'components_per_used_region=1.2500',
                      'native_mass=30.0000', 'native_mass=970.0000'):
            self.assertIn(value, text)
        self.assertIn('does not measure exchanged pooling content', text)

    def test_absent_class_has_undefined_quality_not_perfect_or_zero(self):
        quality = quality_fixture()
        quality['class_2_mass'] = quality['class_2_self_sum'] = 0.
        text = '\n'.join(partition_quality_lines({'sar': quality}, ('sar', 'uniform')))
        self.assertIn('Damaged: same-class=N/A; self_sum=0.0000; native_mass=0.0000', text)
        self.assertIn('uniform: unavailable; missing evidence is not zero.', text)

    def test_actual_diagnostic_accumulator_damaged_transition_semantics(self):
        import torch
        from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator
        truth = torch.tensor([[[2, 2, 1, 3], [0, 0, 1, 3], [0, 0, 0, 0], [0, 0, 0, 0]]])
        hard, sar = truth.clone(), truth.clone()
        hard[0, 0, 0] = 1  # GT Damaged; wrong Intact -> correct Damaged rescue.
        sar[0, 0, 1] = 1   # GT Damaged; correct Damaged -> wrong Intact harm.
        diagnostic = DiagnosticAccumulator(pairs=[('hard', 'sar')])
        diagnostic.update(truth, {'hard': hard, 'sar': sar},
                          (truth > 0).float()[:, None], ['event_test_001'])
        audit = diagnostic.finalize()
        result = audit['groups']['all']['comparisons']['hard__sar']
        self.assertEqual(result['damaged_to_intact_harm'], 1)
        self.assertEqual(result['intact_to_damaged_rescue'], 1)
        text = '\n'.join(_audit_lines(audit, ('hard__sar',)))
        self.assertIn('both condition on GT=Damaged', text)
        self.assertIn('"damaged_to_intact_harm": 1', text)
        self.assertIn('"intact_to_damaged_rescue": 1', text)

    def test_render_reads_quality_totals_from_same_validation_endpoint(self):
        manifest, summary, row = fixture()
        row['partition_quality_totals'] = {name: quality_fixture() for name in GROUPS['support']}
        summary['partition_quality_totals'] = copy.deepcopy(row['partition_quality_totals'])
        row['base_partition_quality_totals'] = copy.deepcopy(row['partition_quality_totals'])
        for quality in row['base_partition_quality_totals'].values():
            quality['class_2_self_sum'] = 1.
        summary['base_partition_quality_totals'] = copy.deepcopy(row['base_partition_quality_totals'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [row])
            text = render(root)
            self.assertIn('Primary completed formal endpoint available: True', text)
            self.assertIn('Unsplit base Q partition quality', text)
            self.assertIn('Read-route after predicted B split partition quality', text)
            self.assertEqual(text.count('Damaged: same-class=10.0000%'), 3)
            self.assertEqual(text.count('Damaged: same-class=20.0000%'), 3)
            self.assertNotIn('Partition-quality totals differ', text)
            summary['partition_quality_totals']['sar']['class_2_self_sum'] = 9.
            write_run(root, manifest, summary, [row])
            text = render(root)
            self.assertIn('Partition-quality totals differ', text)
            self.assertEqual(text.count('Damaged: same-class=20.0000%'), 3)

    def test_new_gate_requires_summary_pass_and_legacy_output_still_works(self):
        for marker, passed, completed in ((True, None, False), (True, False, False),
                                          (True, True, True), (False, None, True)):
            with self.subTest(marker=marker, passed=passed), tempfile.TemporaryDirectory() as temp:
                manifest, summary, row = fixture()
                if marker:
                    manifest['reproduction_gate_version'] = 'history_bounded_v1'
                if passed is not None:
                    summary['baseline_reproduction_pass'] = passed
                root = Path(temp)
                write_run(root, manifest, summary, [row])
                (root / 'reproduction.json').write_text(json.dumps(dict(
                    passed=True, status='passed', metric_checks_passed=True,
                    source_integrity_pass=True, checks=[], warnings=[], policy={})), encoding='utf-8')
                text = render(root)
                self.assertIn(f'Primary completed formal endpoint available: {completed}', text)
                if not completed:
                    self.assertNotIn('Final same-step metrics:', text)

    def test_failed_gate_displays_initial_source_check_without_trained_validation(self):
        manifest, _, row = fixture()
        manifest['reproduction_gate_version'] = 'history_bounded_v1'
        summary = dict(complete=False, suite='support', smoke=False, phase='baseline_check_failed',
                       optimizer_steps=0, test_accessed=False)
        failed = dict(baseline='old_region', expected_baseline='old_region', metric='precision',
                      class_index=2, actual=.503, expected=.5, delta_pp=.3,
                      tolerance_pp=.001, tolerance=1e-5, passed=False,
                      reason='outside bounded tolerance', history=None)
        reproduction = dict(passed=False, status='failed', metric_checks_passed=False,
                            source_integrity_pass=True, checks=[failed], warnings=[],
                            policy={'version': 'history_bounded_v1'})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in (('manifest.json', manifest), ('summary.json', summary),
                                ('initial.json', dict(image_count=450, metrics=row['metrics'])),
                                ('reproduction.json', reproduction)):
                (root / name).write_text(json.dumps(value), encoding='utf-8')
            text = render(root)
            self.assertIn('phase: baseline_check_failed; optimizer_steps: 0', text)
            self.assertIn('Initial source-reproduction evaluation (before training; NOT trained validation)', text)
            self.assertIn('no trained result exists', text)
            self.assertIn('Failed/unverified reproduction checks: 1', text)
            self.assertIn('"class_name": "Damaged"', text)
            self.assertIn('"source_integrity_pass": true', text)
            self.assertNotIn('Latest available validation', text)
            self.assertNotIn('Final same-step metrics:', text)
            self.assertNotIn('validation.jsonl unavailable', text)

    def test_report_shows_historical_checks_even_when_they_pass(self):
        manifest, summary, row = fixture()
        manifest['reproduction_gate_version'] = 'history_bounded_v1'
        summary.update(baseline_reproduction_pass=True, phase='complete', optimizer_steps=5000)
        calibrated = dict(baseline='old_region', expected_baseline='old_region', metric='iou',
                          class_index=3, actual=.70003, expected=.7, delta_pp=.003,
                          tolerance_pp=.005, tolerance=.00005, passed=True, reason='within history tolerance',
                          history={'calibrated': True, 'history_count': 10, 'history_max_abs': .00004})
        reproduction = dict(passed=True, status='passed', metric_checks_passed=True,
                            source_integrity_pass=True, checks=[calibrated],
                            warnings=['Bounded history used'], policy={'history_count': 10})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [row])
            (root / 'reproduction.json').write_text(json.dumps(reproduction), encoding='utf-8')
            text = render(root)
            self.assertIn('Primary completed formal endpoint available: True', text)
            self.assertIn('Checks using expanded historical tolerance: 1', text)
            self.assertIn('"class_name": "Destroyed"', text)
            self.assertIn('"tolerance_pp": 0.005', text)
            self.assertIn('"history": {"calibrated": true, "history_count": 10, "history_max_abs": 4e-05}', text)
            self.assertIn('REPRODUCTION WARNING: Bounded history used', text)

    def test_source_integrity_failure_remains_visible_when_metric_checks_pass(self):
        manifest, summary, row = fixture()
        manifest['reproduction_gate_version'] = 'history_bounded_v1'
        summary.update(complete=False, phase='baseline_check_failed', optimizer_steps=0)
        reproduction = dict(passed=False, status='failed', metric_checks_passed=True,
                            source_integrity_pass=False, checks=[], warnings=[], policy={},
                            error='Source parameters changed')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, manifest, summary, [])
            (root / 'reproduction.json').write_text(json.dumps(reproduction), encoding='utf-8')
            text = render(root)
            self.assertIn('"metric_checks_passed": true', text)
            self.assertIn('"source_integrity_pass": false', text)
            self.assertIn('"passed": false', text)
            self.assertIn('Reproduction failure reason: Source parameters changed', text)
            self.assertIn('Primary completed formal endpoint available: False', text)


if __name__ == '__main__':
    unittest.main()
