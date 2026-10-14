"""Offline evidence-reader tests. No BRIGHT data, model weights, Torch or CUDA."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('BRIGHT_EVIDENCE_AUDIT_V1.py')
spec = importlib.util.spec_from_file_location('evidence_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

JOINT_FORMAT = 'SRAC_REGION_MEMORY_JOINT_TRAIN_V1'

def row(score=0.614, ious=None):
    return {'foreground_miou': score,
            'iou': [.96, .75, .43, .662] if ious is None else ious}

def joint_eval(**changes):
    result = {
        'format': JOINT_FORMAT,
        'image_count': 450,
        'execution_complete': True,
        'whole_val450_processed': True,
        'primary_endpoint': False,
        'full_validation_evidence': False,
        'final_integrity_pending': True,
        'smoke': False, 'test_accessed': False,
        'metrics': {'grid_probq': row(), 'r0': row(), 'rt': row(), 'rf': row()},
    }
    result.update(changes)
    return result

class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'BRIGHT-checkpoints'
        self.base.mkdir()

    def write(self, relative, data):
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return path

    def record(self, report, path):
        return next(r for r in report['records'] if r['source']['path'] == str(path))

    def test_actual_1500_revalidation_is_not_final(self):
        path = self.write('SRAC_JOINT_LIVE_RECOVERY_V2/run_1/train/checkpoint_reval/result.json',
                          joint_eval(final_integrity_pending=False))
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'revalidation')
        self.assertFalse(record['artifact_reverified'])
        self.assertTrue(any('恢复' in n for n in record['notes']))

    def test_actual_2000_pending_integrity_is_intermediate_not_error(self):
        path = self.write('SRAC_JOINT_LIVE_RECOVERY_V2/run_1/train/val_step_002000/result.json', joint_eval())
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'intermediate')
        self.assertEqual(record['step'], 2000)
        self.assertEqual(record['metrics']['rt']['foreground_miou_pct'], 61.4)
        self.assertIn('rt − r0', record['comparisons_pp'])

    def test_user_2000_metrics_preserve_foreground_and_small_deltas(self):
        metrics = {
            'grid_probq': row(.618382, [.967435, .757758, .440493, .656895]),
            'r0': row(.618334, [.968851, .764387, .436566, .654048]),
            'rt': row(.619106, [.968819, .764580, .438297, .654442]),
            'rf': row(.618832, [.968767, .764399, .437885, .654213]),
        }
        path = self.write('SRAC_JOINT_LIVE_RECOVERY_V2/run_1/train/val_step_002000/result.json',
                          joint_eval(metrics=metrics))
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'intermediate')
        self.assertAlmostEqual(record['metrics']['rt']['foreground_miou_pct'], 61.9106)
        # Only rounded input is available; do not fabricate extra hidden precision.
        self.assertAlmostEqual(record['comparisons_pp']['rt − r0'], .0772)
        self.assertAlmostEqual(record['comparisons_pp']['rf − rt'], -.0274)

    def test_new_four_arm_pending_report_is_not_original_three_arm_result(self):
        path = self.write('SRAC_SOFT_MEMORY_JOINT_V1/run_1/train/val_step_002000/result.json',
                          joint_eval(format='SRAC_SOFT_MEMORY_JOINT_V1',
                                     metrics={a: row() for a in ('a', 'b', 'c', 'd')}))
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'intermediate')
        self.assertIn('四组', record['family'])
        self.assertIn('d − b', record['comparisons_pp'])
        self.assertNotIn('rt − r0', record['comparisons_pp'])

    def test_single_run_base_finds_nested_checkpoint_revalidation(self):
        path = self.write('SRAC_JOINT_LIVE_RECOVERY_V2/run_1/train/checkpoint_reval/result.json',
                          joint_eval(final_integrity_pending=False))
        run = self.base / 'SRAC_JOINT_LIVE_RECOVERY_V2/run_1'
        report = audit.collect(run)
        self.assertEqual(report['documents_loaded'], 1)
        self.assertEqual(self.record(report, path)['status'], 'revalidation')

    def test_summary_result_hash_mismatch_downgrades_formal_to_review(self):
        parent = 'SRAC_SAR_FULL_CHAIN_V1/run_1/sar_q/formal/val_baseline/'
        data = {
            'format': 'SRAC_Q_FACTORIAL_EVAL_V1', 'arm': 'baseline',
            'complete': True, 'full_validation_evidence': True, 'image_count': 450,
            'metrics': {'region': row()}, 'immutable': {'files': True},
            'smoke': False, 'test_accessed': False, 'native_a1_replay_ok': True,
            'reproduction': {'passed': True}, 'q_a1_reproduction': {'passed': True},
            'checkpoint_steps': 2000,
        }
        result = self.write(parent + 'result.json', data)
        bound = dict(data, result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
        summary = self.write(parent + 'summary.json', bound)
        report = audit.collect(self.base)
        self.assertEqual(self.record(report, result)['status'], 'formal')
        self.assertEqual(self.record(report, summary)['status'], 'formal')
        self.write(parent + 'summary.json', dict(bound, result_sha256='0' * 64))
        report = audit.collect(self.base)
        for path in (result, summary):
            record = self.record(report, path)
            self.assertEqual(record['status'], 'review')
            self.assertTrue(any('SHA256' in n for n in record['notes']))

    def test_sar_soft_comparison_percentage_cells_are_not_scaled_twice(self):
        cell = {'foreground_miou': 61.4, 'class_iou': [96., 75., 43., 66.2]}
        data = {
            'format': 'SRAC_SAR_SOFT_CROSS_COMPARISON_V1',
            'complete': True, 'full_validation_evidence': True, 'image_count': 450,
            'cells': {
                'baseline': {'unsplit': dict(cell), 'soft': dict(cell)},
                'sar_cue': {'unsplit': dict(cell), 'soft': dict(cell)},
            },
        }
        path = self.write('SRAC_SAR_SOFT_CROSS_V1/run_1/comparison/report.json', data)
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(len(record['metrics']), 4)
        for name in ('baseline/unsplit', 'baseline/soft', 'sar_cue/unsplit', 'sar_cue/soft'):
            self.assertAlmostEqual(record['metrics'][name]['foreground_miou_pct'], 61.4)
            self.assertEqual(record['metrics'][name]['class_iou_pct'], [96., 75., 43., 66.2])

    def test_completed_initial_val_is_not_training_improvement(self):
        path = self.write('SRAC_REGION_MEMORY_JOINT_V1/run_1/train/initial_val/result.json',
                          joint_eval(final_integrity_pending=False))
        record = self.record(audit.collect(self.base), path)
        self.assertNotIn(record['status'], ('formal', 'intermediate', 'train_done'))
        self.assertTrue(any('初始' in n or '训练前' in n or 'initial' in n for n in record['notes']))

    def test_final_flags_without_completed_owner_need_review(self):
        path = self.write('SRAC_REGION_MEMORY_JOINT_V1/run_1/train/final_val/result.json',
                          joint_eval(primary_endpoint=True, full_validation_evidence=True,
                                     final_integrity_pending=False))
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'review')

    def test_mismatched_arm_steps_do_not_pass_formal(self):
        parent = 'SRAC_REGION_MEMORY_JOINT_V1/run_1/train/'
        owner = joint_eval(complete=True, primary_endpoint=True, full_validation_evidence=True,
                           final_integrity_pending=False, updates_per_arm={'r0': 5000, 'rt': 4500, 'rf': 5000},
                           integrity={'files': True}, guard_report={'restored': True, 'failures': []})
        self.write(parent + 'summary.json', owner)
        path = self.write(parent + 'final_val/result.json',
                          joint_eval(primary_endpoint=True, full_validation_evidence=True,
                                     final_integrity_pending=False))
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'review')

    def test_matching_final_owner_is_report_claim_not_rerun(self):
        parent = 'SRAC_REGION_MEMORY_JOINT_V1/run_1/train/'
        owner = joint_eval(complete=True, primary_endpoint=True, full_validation_evidence=True,
                           final_integrity_pending=False, updates_per_arm={'r0': 5000, 'rt': 5000, 'rf': 5000},
                           integrity={'files': True, 'frozen_models': True},
                           guard_report={'restored': True, 'failures': []})
        self.write(parent + 'summary.json', owner)
        path = self.write(parent + 'final_val/result.json',
                          joint_eval(primary_endpoint=True, full_validation_evidence=True,
                                     final_integrity_pending=False, saved_predictions_reopened=True,
                                     final_integrity={'files': True, 'frozen_models': True},
                                     guard_report={'restored': True, 'failures': []}))
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'formal')
        self.assertFalse(record['artifact_reverified'])

    def test_owner_failure_prevents_formal(self):
        parent = 'SRAC_REGION_MEMORY_JOINT_V1/run_1/train/'
        self.write(parent + 'summary.json', joint_eval(
            complete=False, error='CUDA error: device-side assert triggered',
            updates_per_arm={'r0': 1500, 'rt': 1500, 'rf': 1500}))
        path = self.write(parent + 'final_val/result.json',
                          joint_eval(primary_endpoint=True, full_validation_evidence=True,
                                     final_integrity_pending=False))
        record = self.record(audit.collect(self.base), path)
        self.assertNotEqual(record['status'], 'formal')
        self.assertTrue(any('失败' in n for n in record['notes']))

    def test_smoke_execution_complete_false_formal_gate_is_expected(self):
        data = {
            'format': 'SRAC_MEMORY_REBUILD_DIAGNOSE_V3', 'complete': False,
            'execution_complete': True, 'full_validation_evidence': False,
            'image_count': 2, 'smoke': True, 'phase': 'complete_smoke',
            'training': False, 'test_accessed': False,
            'metrics': {'grid_probq': row(.2, [.98, .6, None, None])},
        }
        path = self.write('SRAC_MEMORY_REBUILD_V3/run_1/diagnose_smoke/summary.json', data)
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'smoke')
        self.assertNotEqual(record['status'], 'failed')

    def test_v3_full_labelvote_is_diagnostic_not_reader_result(self):
        data = {
            'format': 'SRAC_MEMORY_REBUILD_DIAGNOSE_V3', 'complete': True,
            'execution_complete': True, 'full_validation_evidence': True,
            'image_count': 450, 'smoke': False, 'training': False,
            'labelvote_is_untrained_diagnostic': True, 'test_accessed': False,
            'validation': {k: True for k in ('protocol', 'bank', 'split', 'labels', 'reference', 'native', 'pool', 'artifacts', 'immutable')},
            'metrics': {'labelvote_t128_top64': row()},
        }
        path = self.write('SRAC_MEMORY_REBUILD_V3/run_1/diagnose/summary.json', data)
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'diagnostic')
        self.assertTrue(any('Reader' in n for n in record['notes']))

    def test_v3_450_processed_without_completion_is_not_formal_diagnostic(self):
        data = {'format': 'SRAC_MEMORY_REBUILD_DIAGNOSE_V3', 'complete': False,
                'execution_complete': True, 'image_count': 450,
                'full_validation_evidence': False, 'phase': 'final_checks'}
        path = self.write('SRAC_MEMORY_REBUILD_V3/run_1/diagnose/summary.json', data)
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'partial')

    def test_sar_failed_train_journal_score_remains_failed(self):
        path = self.write('SRAC_SAR_FULL_CHAIN_V1/run_1/sar_q/formal/train_baseline/summary.json', {
            'format': 'SRAC_Q_FACTORIAL_V1', 'complete': False, 'steps': 2000,
            'error': 'RuntimeError: Frozen A1 parameter or buffer changed',
            'best_valid_miou': .9,
        })
        journal = path.parent / 'validation.jsonl'
        journal.write_text(json.dumps({'step': 2000, 'image_count': 450,
                                     'metrics': {'region': row()}}) + '\n', encoding='utf-8')
        report = audit.collect(self.base)
        scored = self.record(report, journal)
        self.assertEqual(scored['status'], 'failed')
        self.assertAlmostEqual(scored['metrics']['region']['foreground_miou_pct'], 61.4)
        self.assertNotEqual(scored['metrics']['region']['foreground_miou_pct'], 90)

    def test_sar_training_complete_is_separate_from_independent_eval(self):
        data = {'format': 'SRAC_Q_FACTORIAL_V1', 'complete': True, 'steps': 2000,
                'a1_unchanged': True, 'downstream_unchanged': True,
                'full_validation_evidence': True, 'immutable': {'files': True},
                'smoke': False, 'test_accessed': False}
        path = self.write('SRAC_SAR_FULL_CHAIN_V1/run_1/sar_q/formal/train_baseline/summary.json', data)
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'train_done')

    def test_sar_eval_cannot_override_explicit_failed_training(self):
        parent = 'SRAC_SAR_FULL_CHAIN_V1/run_1/sar_q/formal/'
        self.write(parent + 'train_baseline/summary.json', {
            'format': 'SRAC_Q_FACTORIAL_V1', 'complete': False,
            'error': 'Frozen A1 parameter or buffer changed'})
        path = self.write(parent + 'val_baseline/summary.json', {
            'format': 'SRAC_Q_FACTORIAL_EVAL_V1', 'arm': 'baseline',
            'complete': True, 'full_validation_evidence': True, 'image_count': 450,
            'metrics': {'region': row()}, 'immutable': {'files': True},
            'smoke': False, 'test_accessed': False, 'native_a1_replay_ok': True,
            'reproduction': {'passed': True}, 'q_a1_reproduction': {'passed': True},
            'checkpoint_steps': 2000})
        self.assertNotEqual(self.record(audit.collect(self.base), path)['status'], 'formal')

    def test_failed_integrity_blocks_formal(self):
        path = self.write('SRAC_BUILDING_ROUTE_V1/run_1/val/summary.json', {
            'format': 'SRAC_BUILDING_ROUTE_V1_EVAL', 'complete': True,
            'full_validation_evidence': True, 'image_count': 450,
            'immutable': {'files': False}, 'metrics': {'soft': row()}})
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'review')

    def test_test_split_never_enters_val_comparisons(self):
        for suffix, extra in (('success', {}), ('failed', {'error': 'post-eval failure'})):
            path = self.write(f'SRAC_REGION_TEST_V1/run_{suffix}/summary.json', {
                'format': JOINT_FORMAT, 'split': 'test', 'image_count': 878,
                'complete': True, 'metrics': {'r0': row(), 'rt': row()}, **extra})
            record = self.record(audit.collect(self.base), path)
            self.assertEqual(record['metrics'], {})
            self.assertEqual(record['comparisons_pp'], {})

    def test_unknown_format_claiming_formal_is_review(self):
        path = self.write('SRAC_UNKNOWN/run_1/summary.json', {
            'format': 'AN_UNRECOGNIZED_NEW_FORMAT', 'complete': True,
            'full_validation_evidence': True, 'image_count': 450})
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'review')

    def test_truncated_json_is_unreadable_not_absent(self):
        path = self.base / 'SRAC_JOINT/run_1/summary.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"complete": false,', encoding='utf-8')
        record = self.record(audit.collect(self.base), path)
        self.assertEqual(record['status'], 'unreadable')
        self.assertEqual(record['metrics'], {})

    def test_read_size_limit_does_not_read_large_reports(self):
        path = self.write('SRAC_JOINT/run_1/summary.json', {'padding': 'x' * 4096})
        record = self.record(audit.collect(self.base, max_bytes=128), path)
        self.assertEqual(record['status'], 'unreadable')
        self.assertTrue(any('limit' in n for n in record['notes']))

    def test_nan_json_rejected_at_read_boundary(self):
        path = self.base / 'SRAC_JOINT/run_1/summary.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"metrics":{"r0":{"foreground_miou":NaN}}}', encoding='utf-8')
        self.assertEqual(self.record(audit.collect(self.base), path)['status'], 'unreadable')

    def test_journal_ignores_partial_last_line(self):
        path = self.write('SRAC_JOINT/run_1/train/summary.json', {'phase': 'fit'})
        (path.parent / 'train.jsonl').write_text(
            '{"step":1980,"image":"previous"}\n{"step":2000,"image":', encoding='utf-8')
        report = audit.collect(self.base)
        self.assertEqual(report['progress'][0]['step'], 1980)
        self.assertIn('不证明', report['progress'][0]['note'])

    def test_source_files_unchanged_and_save_uses_new_dirs(self):
        path = self.write('SRAC_JOINT/run_1/summary.json', {'phase': 'fit'})
        before = path.read_bytes()
        report = audit.collect(self.base)
        output_parent = self.root / 'New/evidence_records'
        first = audit.save(report, output_parent)
        second = audit.save(report, output_parent)
        self.assertNotEqual(first, second)
        self.assertEqual(path.read_bytes(), before)
        for name in ('REPORT.md', 'evidence.json', 'metrics.csv'):
            self.assertTrue((first / name).is_file())
            self.assertTrue((second / name).is_file())

    def test_cli_rejects_output_inside_checkpoint_tree(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                audit.main(['--base', str(self.base), '--output-parent', str(self.base / 'audit')])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse((self.base / 'audit').exists())

    def test_stdout_mode_does_not_write_output_files(self):
        self.write('SRAC_JOINT/run_1/summary.json', {'phase': 'fit'})
        destination = self.root / 'never-created'
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result = audit.main(['--base', str(self.base), '--output-parent', str(destination), '--stdout-only'])
        self.assertEqual(result, 0)
        self.assertIn('BRIGHT', captured.getvalue())
        self.assertFalse(destination.exists())

    def test_symlink_report_is_not_followed(self):
        target = self.root / 'external.json'
        target.write_text('{"complete": true}', encoding='utf-8')
        link = self.base / 'SRAC_JOINT/run_1/summary.json'
        link.parent.mkdir(parents=True)
        link.symlink_to(target)
        report = audit.collect(self.base)
        self.assertEqual(report['documents_loaded'], 0)
        self.assertTrue(any('符号链接' in w for w in report['warnings']))

    def test_missing_base_is_unknown_not_experiment_failure(self):
        report = audit.collect(self.root / 'missing')
        self.assertEqual(report['records'], [])
        self.assertTrue(report['warnings'])

class MetricTests(unittest.TestCase):
    def test_fraction_to_percent(self):
        result, error = audit.normalize_metric(row())
        self.assertIsNone(error)
        self.assertAlmostEqual(result['foreground_miou_pct'], 61.4)
        self.assertEqual(result['class_iou_pct'], [96., 75., 43., 66.2])

    def test_explicit_percent(self):
        result, error = audit.normalize_metric(
            {'foreground_miou': 61.4, 'iou': [96., 75., 43., 66.2]}, percent=True)
        self.assertIsNone(error)
        self.assertAlmostEqual(result['foreground_miou_pct'], 61.4)

    def test_pct_field(self):
        result, error = audit.normalize_metric(
            {'foreground_miou_pct': 61.4, 'class_iou_pct': [96., 75., 43., 66.2]})
        self.assertIsNone(error)
        self.assertAlmostEqual(result['foreground_miou_pct'], 61.4)

    def test_unknown_percent_in_fraction_field_is_rejected(self):
        result, error = audit.normalize_metric({'foreground_miou': 61.4})
        self.assertIsNone(result)
        self.assertIsNotNone(error)

    def test_unknown_generic_miou_is_not_assumed_foreground(self):
        result, error = audit.normalize_metric({'miou': .72, 'iou': [.96, .75, .43, .662]})
        self.assertIsNone(result)

    def test_nonfinite_boolean_and_out_of_range_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, -0.1, 1.01):
            with self.subTest(value=value):
                result, error = audit.normalize_metric({'foreground_miou': value})
                self.assertIsNone(result)
                self.assertIsNotNone(error)

    def test_mixed_fraction_percent_units_rejected(self):
        result, error = audit.normalize_metric(
            {'foreground_miou_pct': 61.4, 'iou': [.96, .75, .43, .662]})
        self.assertIsNone(result)
        self.assertIsNotNone(error)

    def test_four_class_mean_cannot_be_relabeled_foreground(self):
        result, error = audit.normalize_metric({'foreground_miou': .7005, 'iou': [.96, .75, .43, .662]})
        self.assertIsNone(result)
        self.assertIsNotNone(error)

    def test_inconsistent_confusion_rejected(self):
        data = row()
        data['confusion'] = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        result, error = audit.normalize_metric(data)
        self.assertIsNone(result)
        self.assertIn('混淆', error)

    def test_smoke_absent_classes_can_follow_declared_zero_average(self):
        result, error = audit.normalize_metric(row(.2, [.98, .6, None, None]))
        self.assertIsNone(error)
        self.assertEqual(result['class_iou_pct'], [98., 60., None, None])

class DependencyTests(unittest.TestCase):
    def test_import_and_collect_do_not_require_torch(self):
        code = r'''
import builtins, importlib.util, pathlib, sys, tempfile
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'pandas', 'cv2', 'timm'}:
        raise AssertionError('Forbidden research dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('reader', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as folder:
    m.collect(pathlib.Path(folder))
'''
        result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
