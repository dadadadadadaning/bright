"""Report contracts for fixed contrasts, replay gates and missing evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .report import CASES, CI_PAIRS, FORMAT, PAIRS, PRIMARY, render


def metric(cm):
    tp = [cm[c][c] for c in range(4)]
    support = [sum(row) for row in cm]
    predicted = [sum(row[c] for row in cm) for c in range(4)]

    def ratios(numerator, denominator):
        return [a / b if b else None for a, b in zip(numerator, denominator)]

    iou = ratios(tp, [support[c] + predicted[c] - tp[c] for c in range(4)])
    return dict(confusion=cm, support=support, iou=iou,
                valid_miou=sum(v or 0 for v in iou[1:]) / 3,
                precision=ratios(tp, predicted), recall=ratios(tp, support),
                f1=ratios([2 * v for v in tp], [support[c] + predicted[c] for c in range(4)]),
                background_to_building=sum(cm[0][1:]),
                building_to_background=sum(row[0] for row in cm[1:]))


def fixture():
    base = metric([[70, 10, 15, 5], [5, 80, 10, 5], [5, 10, 80, 5], [5, 5, 10, 80]])
    cosine = metric([[92, 3, 5, 0], [10, 78, 8, 4], [6, 8, 82, 4], [8, 6, 8, 78]])
    metrics = {case: copy.deepcopy(cosine if case == 'cosine' else base) for case in CASES}
    pair_row = dict(count=400, base_correct=310, base_wrong=90, harmed=12, rescued=32,
                    net=20, harm_rate=12 / 310, recovery_rate=32 / 90,
                    damaged_to_intact_harm=2, intact_to_damaged_rescue=4,
                    harm_gt_to_wrong=[[0, 4, 0, 0], [2, 0, 0, 0],
                                      [0, 2, 0, 0], [0, 0, 4, 0]],
                    rescue_gt_from_wrong=[[0, 15, 0, 0], [5, 0, 0, 0],
                                          [0, 4, 0, 0], [0, 0, 8, 0]])
    comparisons = {pair: copy.deepcopy(pair_row) for pair in PAIRS}
    comparisons['normal__identity'].update(rescued=0, harmed=0, net=0)
    bucket = dict(metrics=metrics, comparisons=comparisons)
    reader = dict(queries=8, labelled_queries=6, supported_queries=5,
                  attention_label_queries=4, mass=20., labelled_mass=16.,
                  query_attention_mean=.25, external_attention_mean=.75,
                  external_label_true_mass=.6, cosine_label_true_mass=.7,
                  query_attention_mass_weighted=.4, source_region_accuracy=.5,
                  reader_region_accuracy=.625, changed_region_fraction=.125,
                  attention_by_head=[.2, .3])
    diffs = dict(queries=8, prob_l1=.04, centered_delta_l1=.02, changed_region_fraction=.1)
    summary = dict(format=FORMAT, complete=True, smoke=False, image_count=450,
                   test_accessed=False, source_verified=True, bank_verified=True,
                   head_verified=True, reproduction=dict(passed=True, status='passed'),
                   final_metrics=metrics, identity_passed=True, temperature=.07,
                   replay=dict(max_abs_context=1e-7, max_abs_logits=2e-7,
                               identity_region_argmax_disagreements=0,
                               identity_pixel_argmax_disagreements=0,
                               max_self_weight_error=0., max_external_mass_error=1e-7,
                               max_invalid_weight=0.),
                   routing=dict(active_slots=8, hard_slots=6, tail_slots=2,
                                tail_mass_fraction=.04),
                   reader_stats={case: {group: dict(reader) for group in
                       ('all', 'hard', 'tail', *('Damaged', 'hard__Damaged', 'tail__Damaged'))}
                       for case in ('normal', 'identity', 'uniform', 'cosine')},
                   case_differences={case: dict(diffs, groups={group: dict(diffs) for group in
                       ('all', 'hard', 'tail')}) for case in ('identity', 'uniform', 'cosine')})
    audit = dict(image_count=450, test_accessed=False,
                 groups={name: copy.deepcopy(bucket) for name in
                     ('all', 'boundary', 'foreground_interior', 'roi_true_positive',
                      'roi_false_negative', 'roi_false_positive', 'roi_true_negative',
                      'Damaged__boundary', 'size_small', 'size_medium', 'size_large')},
                 per_event={'event_A': copy.deepcopy(bucket)},
                 per_image=[dict(image=f'image_{i}', event='event_A', metrics=copy.deepcopy(metrics))
                            for i in range(450)])
    return summary, audit


class ReportContracts(unittest.TestCase):
    def write(self, root, summary, audit):
        if summary is not None:
            (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        if audit is not None:
            (root / 'audit_final.json').write_text(json.dumps(audit), encoding='utf-8')

    def render_fixture(self, summary, audit, bootstrap=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, summary, audit)
            return render(root, bootstrap=bootstrap)

    def test_primary_and_both_required_controls_are_fixed(self):
        summary, audit = fixture()
        text = self.render_fixture(summary, audit)
        self.assertIn('PRIMARY cosine minus normal:', text)
        self.assertIn('COMPARE cosine minus nomemory:', text)
        self.assertIn('COMPARE cosine minus uniform:', text)
        self.assertEqual(text.count('PRIMARY '), 1)
        for pair in PAIRS:
            first, second = pair.split('__')
            self.assertIn(f'{second} minus {first}: FG mIoU', text)
        self.assertIn('No ranking or automatic model selection', text)
        self.assertIn('no automatic effectiveness verdict', text)

    def test_render_is_read_only(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, summary, audit)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            render(root)
            after = {path.name: path.read_bytes() for path in root.iterdir()}
        self.assertEqual(before, after)

    def test_identity_and_intervention_protocol_are_explicit(self):
        summary, audit = fixture()
        text = self.render_fixture(summary, audit)
        self.assertIn('normal uses the original production reader path', text)
        self.assertIn('identity manually replays unchanged weights', text)
        self.assertIn('max_abs_context=1e-07', text)
        self.assertIn('max_external_mass_error=1e-07', text)
        self.assertIn('identity_pixel_argmax_disagreements=0', text)
        self.assertIn('Cosine temperature: 0.07', text)
        self.assertIn('no Val temperature sweep', text)
        self.assertIn('Per-query, per-head self weight and total external weight are preserved', text)
        self.assertIn('NOT retrained ablations', text)
        self.assertIn('GT is used only for evaluation', text)
        self.assertIn('independently trained matched-capacity head', text)
        self.assertNotIn('Identity check FAILED', text)

    def test_reproduction_is_compact_with_failed_checks_preserved(self):
        summary, audit = fixture()
        passed = dict(name='normal', metric='iou', class_index=1, passed=True,
                      history={'bulky_history': 'x' * 10000})
        failed = dict(baseline='source_hard', metric='recall', class_index=2,
                      passed=False, delta=.02, tolerance=.00001,
                      reason='absolute difference exceeds tolerance')
        summary['reproduction'] = dict(passed=False, status='failed',
            source=dict(passed=False, checks=[passed, failed], actual='x' * 10000),
            learned=dict(passed=True, status='passed', checks=[passed] * 40,
                         diagnostic_differences=['x' * 10000]))
        text = self.render_fixture(summary, audit)
        self.assertIn('Reproduction / 已训练结果复现: passed=False; status=failed', text)
        self.assertIn('checks=2; passed_checks=1; failed_checks=1; unknown_checks=0', text)
        self.assertIn('checks=40; passed_checks=40; failed_checks=0; unknown_checks=0', text)
        self.assertIn('FAIL source_hard/recall/2: delta=0.02; tolerance=1e-05;', text)
        self.assertIn('absolute difference exceeds tolerance', text)
        self.assertIn('reproduction.json', text)
        self.assertNotIn('bulky_history', text)
        self.assertNotIn('diagnostic_differences', text)
        self.assertNotIn('x' * 100, text)

    def test_class_counts_and_paired_foreground_boundary_roi_size_events(self):
        summary, audit = fixture()
        text = self.render_fixture(summary, audit)
        self.assertIn('Intact: dTP=-2; dFP=-8; dFN=2', text)
        self.assertIn('Destroyed: dTP=-2; dFP=-7; dFN=2', text)
        self.assertIn('Damaged FP change by true class: Background=-10; Intact=-2; Destroyed=-2', text)
        self.assertIn('TP=82; FP=21; FN=18', text)
        self.assertIn('rescued=32; harmed=12; net=20', text)
        self.assertIn('GT foreground (all GT>0): rescued=17; harmed=8; net=9', text)
        self.assertIn('GT building-side boundary: rescued=17; harmed=8; net=9', text)
        for group in ('roi_true_positive', 'roi_false_negative', 'roi_false_positive',
                      'roi_true_negative', 'size_small', 'size_medium', 'size_large'):
            self.assertIn(group + ': count=400', text)
        self.assertIn('Damaged->Intact harm=2; Intact->Damaged rescue=4', text)
        self.assertIn('event_A: delta=', text)
        self.assertIn('GT support B/I/D/X=100/100/100/100', text)
        self.assertIn('conditional scores are NOT full-image mIoU', text)
        self.assertIn('NOT building instances', text)
        self.assertNotIn('image_449', text)

    def test_four_reader_cases_and_difference_groups_are_retained(self):
        summary, audit = fixture()
        text = self.render_fixture(summary, audit)
        for case in ('normal', 'identity', 'uniform', 'cosine'):
            self.assertIn(f'Reader {case}:', text)
        self.assertIn('tail__Damaged | 8 | 6 | 5 | 4', text)
        self.assertIn('all attention_by_head: 0.2, 0.3', text)
        self.assertIn('tail mass-weighted: mass=20; labelled_mass=16; query_attention=0.4', text)
        self.assertIn('Intervention cosine versus normal: queries=8; prob_l1=0.04;', text)
        self.assertIn('tail: queries=8; prob_l1=0.04;', text)
        self.assertIn('not Q-mass-weighted accuracy or pixel accuracy', text)
        self.assertIn('label mass is NOT classification accuracy', text)
        self.assertIn('NOT causal contribution', text)
        self.assertIn('attention_label_queries', text)

    def test_wrapped_reader_schema_and_null_empty_groups(self):
        summary, audit = fixture()
        summary['reader_stats'] = {'cases': summary['reader_stats']}
        summary['reader_stats']['cases']['cosine']['tail'] = dict(
            queries=0, labelled_queries=0, supported_queries=0, attention_label_queries=0,
            external_label_true_mass=None, cosine_label_true_mass=None,
            query_attention_mean=None, attention_by_head=[None, None])
        text = self.render_fixture(summary, audit)
        self.assertIn('tail | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A', text)
        self.assertIn('tail attention_by_head: N/A, N/A', text)

    def test_missing_files_do_not_create_zero_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            text = render(directory, bootstrap=5)
        self.assertIn('summary.json: unavailable', text)
        self.assertIn('audit_final.json: unavailable', text)
        self.assertIn('cosine: FG mIoU=N/A', text)
        self.assertIn('PRIMARY cosine minus normal: FG mIoU N/A', text)
        self.assertIn('GT foreground (all GT>0): rescued=N/A; harmed=N/A; net=N/A', text)
        self.assertIn('Smoke or unknown smoke status', text)
        self.assertIn('Identity check FAILED or unavailable', text)
        self.assertIn('Bootstrap unavailable', text)
        self.assertNotIn('+0.0000 pp', text)

    def test_missing_pair_and_subset_are_not_zero(self):
        summary, audit = fixture()
        audit['groups']['all']['comparisons'].pop(PRIMARY)
        audit['groups'].pop('roi_false_negative')
        text = self.render_fixture(summary, audit)
        line = next(line for line in text.splitlines() if line.startswith('PRIMARY '))
        self.assertIn('net=N/A', line)
        self.assertIn('normal__cosine: paired evidence unavailable', text)
        self.assertIn('roi_false_negative: count=N/A', text)

    def test_missing_or_invalid_paired_transition_cannot_be_inferred_from_cm(self):
        for bad in (None, [[0]], [[0, 0, 0, 0]] * 3, [[0, 0, -1, 0]] * 4):
            with self.subTest(bad=bad):
                summary, audit = fixture()
                row = audit['groups']['all']['comparisons'][PRIMARY]
                row['rescue_gt_from_wrong'] = bad
                text = self.render_fixture(summary, audit)
                self.assertIn('GT foreground (all GT>0): rescued=N/A; harmed=8; net=N/A', text)

    def test_truncated_json_and_null_rows_remain_readable(self):
        summary, _ = fixture()
        summary.update(complete=False, smoke=True, reader_stats=None, case_differences=None,
                       final_metrics={'normal': None, 'cosine': dict(valid_miou=None, iou=[None] * 4)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, summary, None)
            (root / 'audit_final.json').write_text('{"image_count":', encoding='utf-8')
            text = render(root, bootstrap=5)
        self.assertIn('audit_final.json: unreadable', text)
        self.assertIn('Incomplete run', text)
        self.assertIn('execution evidence only, NOT final performance', text)
        self.assertIn('cosine: FG mIoU=N/A', text)

    def test_bootstrap_reports_three_predeclared_comparisons(self):
        summary, audit = fixture()
        with patch('bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap',
                   side_effect=lambda rows, pair, repetitions: {
                       'pair': pair, 'repetitions': repetitions, 'percentile_95_pp': [-.1, .1]}) as call:
            text = self.render_fixture(summary, audit, bootstrap=20)
        self.assertEqual([c.args[1:] for c in call.call_args_list], [(pair, 20) for pair in CI_PAIRS])
        self.assertEqual(text.count('Paired image/event-stratified bootstrap:'), 3)
        self.assertIn('not seed variance or unseen-event generalization', text)
        self.assertIn('spatial tile dependence unresolved', text)
        self.assertIn('intervals are pointwise', text)

    def test_identity_boolean_does_not_override_prediction_disagreement(self):
        for field in ('identity_region_argmax_disagreements', 'identity_pixel_argmax_disagreements'):
            with self.subTest(field=field):
                summary, audit = fixture()
                summary['replay'][field] = 1
                with patch('bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap') as call:
                    text = self.render_fixture(summary, audit, bootstrap=5)
                call.assert_not_called()
                self.assertIn('Identity check FAILED or unavailable', text)
                self.assertIn('不能将差异解释为邻居权重干预效果', text)

    def test_smoke_provenance_identity_and_reproduction_gate_all_intervals(self):
        summary, audit = fixture()
        changes = [dict(smoke=True), dict(smoke=None), dict(complete=False),
                   dict(format='MEMORY_READOUT_AUDIT_V1'), dict(image_count=449),
                   dict(source_verified=False), dict(bank_verified=False), dict(head_verified=False),
                   dict(test_accessed=True), dict(identity_passed=False), dict(replay={}),
                   dict(temperature=.1), dict(temperature=None),
                   dict(reproduction={'passed': False, 'status': 'passed'})]
        for change in changes:
            with self.subTest(change=change), patch(
                    'bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap') as call:
                text = self.render_fixture(dict(summary, **change), audit, bootstrap=5)
                call.assert_not_called()
                self.assertEqual(text.count('Bootstrap unavailable ('), 3)

    def test_incomplete_audit_records_gate_bootstrap(self):
        summary, audit = fixture()
        for change in (dict(image_count=449), dict(test_accessed=True),
                       dict(per_image=[]), dict(per_image=None)):
            with self.subTest(change=change), patch(
                    'bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap') as call:
                text = self.render_fixture(summary, dict(audit, **change), bootstrap=5)
                call.assert_not_called()
                self.assertIn('Bootstrap unavailable', text)

    def test_missing_bootstrap_metrics_fail_without_fabricating_interval(self):
        summary, audit = fixture()
        audit['per_image'][0]['metrics'].pop('cosine')
        text = self.render_fixture(summary, audit, bootstrap=2)
        self.assertIn('invalid/missing paired evidence (KeyError)', text)
        self.assertNotIn('percentile_95_pp', text)

    def test_actual_bootstrap_keeps_pair_direction(self):
        summary, audit = fixture()
        text = self.render_fixture(summary, audit, bootstrap=3)
        records = [json.loads(line.split(': ', 1)[1]) for line in text.splitlines()
                   if line.startswith('Paired image/event-stratified bootstrap:')]
        self.assertEqual([record['pair'] for record in records], list(CI_PAIRS))
        expected = 100 * (summary['final_metrics']['cosine']['valid_miou'] -
                          summary['final_metrics']['normal']['valid_miou'])
        self.assertAlmostEqual(records[0]['delta_pp'], expected)
        self.assertEqual(records[0]['image_count'], 450)

    def test_no_bootstrap_by_default_and_negative_count_rejected(self):
        summary, audit = fixture()
        with patch('bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap') as call:
            self.render_fixture(summary, audit)
        call.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            render('missing', bootstrap=-1)


if __name__ == '__main__':
    unittest.main()
