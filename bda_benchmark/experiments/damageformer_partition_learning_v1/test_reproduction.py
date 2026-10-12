"""Pure-standard-library regression tests for the pre-training baseline gate."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .reproduction import (
    DEFAULT_TOL, MAX_TOL, build_policy, compare_baselines, enforce, load_history,
)


def metric():
    return dict(valid_miou=.5, iou=[.9, .6, .4, .5],
                precision=[.9, .7, .6, .7], recall=[.9, .7, .6, .7],
                count=100, support=[25, 25, 25, 25],
                confusion=[[25, 0, 0, 0], [0, 25, 0, 0], [0, 0, 25, 0], [0, 0, 0, 25]])


def fixtures():
    expected = {name: metric() for name in ('old_pixel', 'old_region', 'hard')}
    actual = {name: metric() for name in ('old_pixel', 'old_region', 'source_hard')}
    rows = [dict(step=step, image_count=450, test_accessed=False,
                 metrics=copy.deepcopy(expected)) for step in (500, 1000, 1500)]
    return actual, expected, rows


def check(report, baseline, field, index=None):
    return next(row for row in report['checks'] if row['baseline'] == baseline and
                row['metric'] == field and row['class_index'] == index)


class BaselineReproductionTests(unittest.TestCase):
    def setUp(self):
        self.actual, self.expected, self.rows = fixtures()

    def compare(self, policy=None):
        return compare_baselines(self.actual, self.expected,
                                 policy or build_policy(self.expected, self.rows))

    def test_exact_reproduction_and_enforcement(self):
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['checks']), 45)
        self.assertIsNone(enforce(report))
        json.dumps(report, allow_nan=False)

    def test_observed_history_calibrates_only_the_affected_region_cell(self):
        self.rows[0]['metrics']['old_region']['precision'][2] += .00001507
        self.actual['old_region']['precision'][2] += .000022
        policy = build_policy(self.expected, self.rows)
        cell = policy['entries']['old_region/precision/2']
        self.assertAlmostEqual(cell['tolerance'], .00002507)
        self.assertTrue(self.compare(policy)['passed'])
        self.assertEqual(policy['entries']['old_region/precision/1']['tolerance'], DEFAULT_TOL)
        self.assertEqual(policy['entries']['old_pixel/precision/2']['tolerance'], DEFAULT_TOL)

    def test_history_within_default_does_not_add_a_margin(self):
        self.rows[0]['metrics']['old_region']['recall'][2] -= .000009
        policy = build_policy(self.expected, self.rows)
        self.assertEqual(policy['entries']['old_region/recall/2']['tolerance'], DEFAULT_TOL)
        self.actual['old_region']['recall'][2] += .000015
        self.assertFalse(self.compare(policy)['passed'])

    def test_cap_applies_to_current_deviation(self):
        self.rows[0]['metrics']['old_region']['recall'][2] -= .000045
        policy = build_policy(self.expected, self.rows)
        self.assertEqual(policy['entries']['old_region/recall/2']['tolerance'], MAX_TOL)
        self.actual['old_region']['recall'][2] += .000049
        self.assertTrue(self.compare(policy)['passed'])
        self.actual['old_region']['recall'][2] += .000002
        self.assertFalse(self.compare(policy)['passed'])

    def test_anomalous_history_blocks_even_an_exact_current_result(self):
        self.rows[0]['metrics']['old_region']['iou'][2] -= .000051
        policy = build_policy(self.expected, self.rows)
        self.assertEqual(policy['entries']['old_region/iou/2']['tolerance'], MAX_TOL)
        self.assertTrue(policy['entries']['old_region/iou/2']['history_anomaly'])
        self.assertFalse(self.compare(policy)['passed'])

    def test_pixel_and_hard_always_retain_strict_tolerances(self):
        for baseline, source in (('old_pixel', 'old_pixel'), ('source_hard', 'hard')):
            with self.subTest(baseline=baseline):
                self.rows[0]['metrics'][source]['iou'][2] -= .1
                policy = build_policy(self.expected, self.rows)
                self.actual[baseline]['iou'][2] += .000011
                self.assertFalse(check(self.compare(policy), baseline, 'iou', 2)['passed'])
                self.actual[baseline] = metric()

    def test_current_actual_cannot_change_policy_or_source_history(self):
        self.rows[0]['metrics']['old_region']['recall'][2] -= .00002
        before = copy.deepcopy((self.expected, self.rows))
        policy = build_policy(self.expected, self.rows)
        frozen = copy.deepcopy(policy)
        self.actual['old_region']['recall'][2] += .2
        report = self.compare(policy)
        self.assertFalse(report['passed'])
        self.assertEqual(policy, frozen)
        self.assertEqual((self.expected, self.rows), before)
        self.actual['old_region']['recall'][2] = .6
        self.assertNotEqual(report['actual'], self.actual)

    def test_comparison_collects_all_failures(self):
        self.actual['old_pixel']['precision'][1] += .01
        self.actual['old_region']['recall'][2] -= .01
        self.actual['source_hard']['iou'][3] += .01
        report = self.compare()
        self.assertEqual(sum(not row['passed'] for row in report['checks']), 3)
        with self.assertRaisesRegex(ValueError, '3 checks'):
            enforce(report)

    def test_missing_and_wrong_shapes_fail_closed(self):
        cases = [None, [], [.5]*3, [.5]*5, {'0': .5}, (.5,)*4]
        for value in cases:
            with self.subTest(value=value):
                self.actual['old_region']['precision'] = value
                report = self.compare()
                self.assertFalse(report['passed'])
                json.dumps(report, allow_nan=False)
        del self.actual['old_region']['precision']
        self.assertFalse(self.compare()['passed'])
        self.actual['old_region'] = metric()
        self.actual['old_region']['valid_miou'] = [.5]
        self.assertFalse(self.compare()['passed'])

    def test_missing_baselines_and_invalid_top_level_fail_closed(self):
        del self.actual['source_hard']
        self.assertFalse(self.compare()['passed'])
        self.assertFalse(compare_baselines(None, self.expected, build_policy(self.expected))['passed'])
        self.assertFalse(compare_baselines(self.actual, None, build_policy(None))['passed'])

    def test_malformed_expected_subrecord_does_not_discard_the_report(self):
        self.expected['hard'] = None
        policy = build_policy(self.expected, self.rows)
        report = self.compare(policy)
        self.assertFalse(report['passed'])
        self.assertTrue(any(row['baseline'] == 'source_hard' and not row['passed']
                            for row in report['checks']))
        json.dumps(report, allow_nan=False)

    def test_none_semantics_are_preserved(self):
        for data in (self.actual, self.expected):
            data['old_region']['precision'][2] = None
        for row in self.rows:
            row['metrics']['old_region']['precision'][2] = None
        self.assertTrue(self.compare()['passed'])
        self.actual['old_region']['precision'][2] = 0.
        self.assertFalse(self.compare()['passed'])

    def test_nonfinite_values_are_rejected_and_preserved_as_json_safe_evidence(self):
        for value, representation in ((float('nan'), 'NaN'), (float('inf'), 'Infinity'),
                                      (-float('inf'), '-Infinity')):
            with self.subTest(value=representation):
                self.actual['old_region']['precision'][2] = value
                report = self.compare()
                self.assertFalse(report['passed'])
                self.assertEqual(check(report, 'old_region', 'precision', 2)['actual'], representation)
                json.dumps(report, allow_nan=False)
        self.expected['old_pixel']['recall'][1] = float('nan')
        report = self.compare()
        self.assertFalse(report['passed'])
        json.dumps(report, allow_nan=False)

    def test_bool_strings_and_out_of_range_metrics_fail(self):
        for value in (True, '0.6', -0.1, 1.1):
            self.actual['old_region']['precision'][2] = value
            self.assertFalse(self.compare()['passed'])

    def test_count_and_support_must_match_but_confusion_need_not(self):
        self.actual['old_region']['confusion'][0][0] -= 1
        self.actual['old_region']['confusion'][0][1] += 1
        self.assertTrue(self.compare()['passed'])
        self.actual['old_region']['support'] = [24, 26, 25, 25]
        self.assertFalse(check(self.compare(), 'old_region', 'support')['passed'])
        self.actual['old_region']['support'] = [25]*4
        self.actual['old_region']['count'] = 101
        self.assertFalse(check(self.compare(), 'old_region', 'count')['passed'])

    def test_optional_metadata_can_be_absent_on_both_sides(self):
        for metric_set in (self.actual, self.expected):
            for record in metric_set.values():
                record.pop('count')
                record.pop('support')
        report = compare_baselines(self.actual, self.expected, build_policy(self.expected))
        self.assertTrue(report['passed'])
        self.actual['old_region']['count'] = True
        self.assertFalse(compare_baselines(self.actual, self.expected, build_policy(self.expected))['passed'])

    def test_partial_metadata_and_inconsistent_total_fail(self):
        del self.actual['old_region']['support']
        self.assertFalse(self.compare()['passed'])
        self.actual['old_region'] = metric()
        self.expected['old_region']['count'] = 99
        self.actual['old_region']['count'] = 99
        self.assertFalse(self.compare()['passed'])

    def test_too_short_history_retains_defaults(self):
        for rows in ([], self.rows[:2]):
            policy = build_policy(self.expected, rows)
            self.assertFalse(policy['history_usable'])
            self.assertTrue(policy['warnings'])
            self.assertTrue(all(row['tolerance'] == DEFAULT_TOL for row in policy['entries'].values()))

    def test_bad_history_record_disables_all_calibration(self):
        self.rows[0]['metrics']['old_region']['precision'][2] += .000025
        defects = [('image_count', 2), ('test_accessed', True), ('step', 500)]
        for field, value in defects:
            rows = copy.deepcopy(self.rows)
            rows[1][field] = value
            policy = build_policy(self.expected, rows)
            self.assertFalse(policy['history_usable'])
            self.assertEqual(policy['entries']['old_region/precision/2']['tolerance'], DEFAULT_TOL)
        rows = copy.deepcopy(self.rows)
        rows[1]['metrics']['old_region']['precision'][2] = None
        self.assertFalse(build_policy(self.expected, rows)['history_usable'])
        rows[1]['metrics']['old_region']['precision'][2] = float('nan')
        policy = build_policy(self.expected, rows)
        self.assertFalse(policy['history_usable'])
        json.dumps(policy, allow_nan=False)

    def test_changed_historical_gt_metadata_disables_calibration(self):
        self.rows[0]['metrics']['old_region']['precision'][2] += .000025
        self.rows[1]['metrics']['old_region']['support'] = [24, 26, 25, 25]
        self.assertFalse(build_policy(self.expected, self.rows)['history_usable'])

    def test_policy_cannot_broaden_hard_or_exceed_cap(self):
        for key, value in (('source_hard/iou/1', .00002), ('old_region/iou/1', .1)):
            policy = build_policy(self.expected, self.rows)
            policy['entries'][key]['tolerance'] = value
            self.assertFalse(self.compare(policy)['passed'])

    def test_load_history_returns_hash_and_does_not_change_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'validation.jsonl'
            content = ''.join(json.dumps(row)+'\n' for row in self.rows).encode()
            path.write_bytes(content)
            rows, warnings, sha = load_history(root, 5000)
            self.assertEqual(rows, self.rows)
            self.assertEqual(warnings, [])
            self.assertEqual(sha, hashlib.sha256(content).hexdigest())
            self.assertEqual(path.read_bytes(), content)

    def test_missing_and_truncated_history_fall_back(self):
        with tempfile.TemporaryDirectory() as root:
            rows, warnings, sha = load_history(root, 5000)
            self.assertEqual(rows, [])
            self.assertTrue(warnings)
            self.assertIsNone(sha)
            path = Path(root)/'validation.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows)+'{"step":', encoding='utf-8')
            rows, warnings, sha = load_history(root, 5000)
            self.assertEqual(rows, [])
            self.assertTrue(warnings)
            self.assertIsNotNone(sha)

    def test_history_rejects_future_steps_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'validation.jsonl'
            clean = ''.join(json.dumps(row)+'\n' for row in self.rows)
            for text, final in ((clean, 1000), (clean+'{"step": 1, "step": 2}\n', 5000),
                                (clean.replace('0.6', 'NaN', 1), 5000)):
                with self.subTest(final=final, suffix=text[-30:]):
                    path.write_text(text, encoding='utf-8')
                    rows, warnings, sha = load_history(root, final)
                    self.assertEqual(rows, [])
                    self.assertTrue(warnings)
                    self.assertIsNotNone(sha)

    def test_history_boundary_uses_exact_default_for_zero_reference(self):
        self.expected['old_region']['valid_miou'] = 0.
        self.actual['old_region']['valid_miou'] = DEFAULT_TOL
        for row in self.rows:
            row['metrics']['old_region']['valid_miou'] = DEFAULT_TOL
        policy = build_policy(self.expected, self.rows)
        self.assertEqual(policy['entries']['old_region/valid_miou']['tolerance'], DEFAULT_TOL)
        self.assertTrue(self.compare(policy)['passed'])


if __name__ == '__main__':
    unittest.main()
