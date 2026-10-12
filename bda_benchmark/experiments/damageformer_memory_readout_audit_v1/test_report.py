"""The report must distinguish missing evidence, matched controls and smoke runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .report import CASES, FORMAT, PAIRS, PRIMARY, render


def metric(cm):
    tp = [cm[c][c] for c in range(4)]
    support = [sum(row) for row in cm]
    predicted = [sum(row[c] for row in cm) for c in range(4)]
    def ratio(a, b):
        return [x / y if y else None for x, y in zip(a, b)]
    iou = ratio(tp, [support[c] + predicted[c] - tp[c] for c in range(4)])
    return dict(confusion=cm, support=support,
                valid_miou=sum(v or 0 for v in iou[1:]) / 3,
                iou=iou, precision=ratio(tp, predicted), recall=ratio(tp, support),
                f1=ratio([2*v for v in tp], [support[c] + predicted[c] for c in range(4)]),
                background_to_building=sum(cm[0][1:]),
                building_to_background=sum(row[0] for row in cm[1:]))


def fixture():
    base = metric([[70, 10, 15, 5], [5, 80, 10, 5], [5, 10, 80, 5], [5, 5, 10, 80]])
    normal = metric([[92, 3, 5, 0], [10, 78, 8, 4], [6, 8, 82, 4], [8, 6, 8, 78]])
    metrics = {case: copy.deepcopy(normal if case == 'normal' else base) for case in CASES}
    pair_row = dict(count=400, base_correct=310, base_wrong=90, harmed=12, rescued=32,
                    net=20, harm_rate=12/310, recovery_rate=32/90,
                    damaged_to_intact_harm=2, intact_to_damaged_rescue=4)
    bucket = dict(metrics=metrics, comparisons={pair: dict(pair_row) for pair in PAIRS})
    reader_row = dict(queries=8, labelled_queries=6, supported_queries=5, attention_label_queries=4,
                      mass=20., labelled_mass=16., query_attention_mean=.25,
                      query_attention_mass_weighted=.4,
                      external_attention_mean=.75, external_label_true_mass=.6,
                      cosine_label_true_mass=.7, same_class_external_attention_mean=.45,
                      delta_abs_mean=.1, centered_delta_abs_mean=.05, prob_l1_mean=.03,
                      changed_region_fraction=.125, source_region_accuracy=.5,
                      reader_region_accuracy=.625, attention_by_head=[.2, .3])
    summary = dict(format=FORMAT, complete=True, smoke=False, image_count=450,
                   test_accessed=False, source_verified=True, bank_verified=True,
                   head_verified=True, reproduction=dict(passed=True), final_metrics=metrics,
                   replay=dict(max_abs_context=1e-7, max_abs_logits=2e-7, argmax_disagreements=0),
                   routing=dict(active_slots=8, hard_slots=6, tail_slots=2,
                                tail_mass_fraction=.04, mean_pixel_tail_fraction=.04),
                   reader_stats=dict(cases={'normal': {k: dict(reader_row) for k in
                       ('all', 'hard', 'tail', 'Damaged', 'hard__Damaged', 'tail__Damaged')}}),
                   case_differences={'off': dict(prob_l1=.4, centered_delta_l1=.2,
                                                changed_region_fraction=.1)})
    audit = dict(image_count=450, test_accessed=False,
                 groups={name: copy.deepcopy(bucket) for name in
                         ('all', 'boundary', 'foreground_interior', 'Damaged__boundary')},
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

    def test_complete_primary_and_same_weight_interventions(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            text = render(root, pair='source_hard__normal')
            after = {p.name: p.read_bytes() for p in root.iterdir()}
        self.assertEqual(before, after)
        self.assertIn('PRIMARY normal minus nomemory:', text)
        self.assertIn('COMPARE normal minus source_hard:', text)
        self.assertIn('COMPARE shuffle_labels minus normal:', text)
        self.assertIn('COMPARE off minus normal:', text)
        self.assertIn('COMPARE hard_only minus source_hard:', text)
        self.assertIn('COMPARE tail_only minus source_hard:', text)
        self.assertIn('COMPARE normal minus hard_only:', text)
        self.assertIn('COMPARE normal minus tail_only:', text)
        self.assertIn('NOT retrained ablations', text)
        self.assertIn('off is NOT nomemory or source_hard', text)
        self.assertIn('NOT historical A1', text)
        self.assertIn('no automatic effectiveness verdict', text)

    def test_classes_net_errors_foreground_boundary_and_event_survive(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            text = render(root)
        self.assertIn('Intact: dTP=-2; dFP=-8; dFN=2', text)
        self.assertIn('Destroyed: dTP=-2; dFP=-7; dFN=2', text)
        self.assertIn('Damaged FP change by true class: Background=-10; Intact=-2; Destroyed=-2', text)
        self.assertIn('rescued=32; harmed=12; net=20', text)
        self.assertIn('Damaged->Intact harm=2; Intact->Damaged rescue=4', text)
        self.assertIn('foreground_interior: count=400', text)
        self.assertIn('Damaged__boundary: count=400', text)
        self.assertIn('event_A: delta=', text)
        self.assertIn('GT support B/I/D/X=100/100/100/100', text)
        self.assertIn('conditional scores are NOT full-image mIoU', text)
        self.assertNotIn('image_449', text)

    def test_actual_attention_label_only_posterior_and_mass_are_distinguished(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            text = render(root)
        self.assertIn('Reader normal:', text)
        self.assertIn('tail__Damaged | 8 | 6', text)
        self.assertIn('all attention_by_head: 0.2, 0.3', text)
        self.assertIn('not Q-mass-weighted accuracy', text)
        self.assertIn('tail mass-weighted: mass=20; labelled_mass=16; query_attention=0.4', text)
        self.assertIn('nonzero external attention (attention_label_queries)', text)
        self.assertIn('query self-token attention per head', text)
        self.assertIn('cosine posterior is separate label-only evidence', text)
        self.assertIn('NOT causal contribution', text)
        self.assertIn('Intervention off versus normal: prob_l1=0.4; centered_delta_l1=0.2; changed_region_fraction=0.1', text)
        self.assertIn('Intervention shuffle_labels versus normal: prob_l1=N/A', text)

    def test_missing_files_are_not_zero_and_bootstrap_cannot_hide_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            text = render(temp, bootstrap=10)
        self.assertIn('summary.json: unavailable', text)
        self.assertIn('audit_final.json: unavailable', text)
        self.assertIn('normal: FG mIoU=N/A', text)
        self.assertIn('PRIMARY normal minus nomemory: FG mIoU N/A', text)
        self.assertIn('paired evidence unavailable', text)
        self.assertIn('Smoke or unknown smoke status', text)
        self.assertIn('Bootstrap unavailable', text)
        self.assertNotIn('+0.0000 pp', text)

    def test_direct_reader_stats_schema_preserves_empty_group_nulls(self):
        summary, audit = fixture()
        summary['reader_stats'] = summary['reader_stats']['cases']
        summary['reader_stats']['normal']['tail'] = dict(
            queries=0, labelled_queries=0, supported_queries=0, attention_label_queries=0,
            external_label_true_mass=None, cosine_label_true_mass=None,
            query_attention_mean=None, attention_by_head=[None, None])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            text = render(root)
        self.assertIn('tail | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A', text)
        self.assertIn('tail attention_by_head: N/A, N/A', text)
        self.assertIn('Reader normal:', text)

    def test_tail_pixel_bins_keep_conditional_denominators_and_pair_direction(self):
        summary, audit = fixture()
        summary['replay']['observation_pixel_argmax_disagreements'] = 3
        bins = {key + suffix: copy.deepcopy(audit['groups']['all'])
                for key in ('zero', 'positive_le_01', 'gt_01_le_05', 'gt_05')
                for suffix in ('', '__foreground')}
        bins['gt_05']['comparisons']['hard_only__normal'].update(
            count=12, rescued=2, harmed=5, net=-3)
        del bins['zero__foreground']['comparisons'][PRIMARY]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            (root / 'tail_pixel_bins.json').write_text(json.dumps(bins), encoding='utf-8')
            text = render(root)
        self.assertIn('observation_pixel_argmax_disagreements=3', text)
        self.assertIn('actual predictions remain on the original need_weights=False path', text)
        self.assertIn('NOT attention or GT purity', text)
        self.assertIn('Fixed bin thresholds / 固定门槛: [0, 0.1, 0.5, 1]', text)
        self.assertIn('all valid pixels tail fraction (0.5,1]:', text)
        self.assertIn('GT foreground pixels tail fraction (0,0.1]:', text)
        self.assertIn('normal minus hard_only: count=12;', text)
        self.assertIn('rescued=2; harmed=5; net=-3', text)
        self.assertIn('normal minus nomemory: count=N/A;', text)
        self.assertNotIn('hard_only minus normal:', text)

    def test_missing_pixel_bin_file_has_no_synthetic_counts(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            text = render(root)
        self.assertIn('tail_pixel_bins.json: unavailable', text)
        self.assertIn('Pixel-bin paired evidence unavailable', text)

    def test_truncated_json_and_null_metrics_remain_readable(self):
        summary, _ = fixture()
        summary.update(complete=False, smoke=True)
        summary['final_metrics'] = {'normal': dict(valid_miou=None, iou=[None]*4)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, None)
            (root / 'audit_final.json').write_text('{"image_count":', encoding='utf-8')
            text = render(root, pair='missing__pair', bootstrap=10)
        self.assertIn('audit_final.json: unreadable', text)
        self.assertIn('Incomplete run', text)
        self.assertIn('execution evidence only, NOT final performance', text)
        self.assertIn('Requested pair missing__pair unavailable; not treated as zero.', text)
        self.assertIn('normal: FG mIoU=N/A', text)

    def test_bootstrap_uses_primary_and_fixed_event_scope(self):
        summary, audit = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            with patch('bda_benchmark.experiments.damageformer_roi_followup_v1.report.paired_bootstrap',
                       return_value={'pair': PRIMARY, 'percentile_95_pp': [1., 2.]}) as bootstrap:
                text = render(root, bootstrap=20)
                self.assertEqual(bootstrap.call_args.args[1:], (PRIMARY, 20))
        self.assertIn('Paired image/event-stratified bootstrap', text)
        self.assertIn('not seed variance or unseen-event generalization', text)

    def test_smoke_unverified_reproduction_and_missing_rows_block_bootstrap(self):
        summary, audit = fixture()
        changes = [dict(smoke=True), dict(complete=False), dict(source_verified=False),
                   dict(bank_verified=False), dict(head_verified=False), dict(test_accessed=True),
                   dict(reproduction={'passed': False, 'status': 'passed'}), dict(image_count=449)]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = dict(summary, **change)
                self.write(root, data, audit)
                self.assertIn('Bootstrap unavailable', render(root, bootstrap=5))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit['per_image'] = audit['per_image'][:2]
            self.write(root, summary, audit)
            self.assertIn('Bootstrap unavailable', render(root, bootstrap=5))

    def test_absent_pair_has_no_manufactured_net_errors(self):
        summary, audit = fixture()
        audit['groups']['all']['comparisons'].pop(PRIMARY)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, summary, audit)
            text = render(root)
        primary_line = next(line for line in text.splitlines() if line.startswith('PRIMARY '))
        self.assertIn('net=N/A', primary_line)
        self.assertIn('nomemory__normal: paired evidence unavailable', text)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            render('missing', bootstrap=-1)
        for pair in ('normal', 'a__b__c', 'normal__normal'):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                render('missing', pair=pair)


if __name__ == '__main__':
    unittest.main()
