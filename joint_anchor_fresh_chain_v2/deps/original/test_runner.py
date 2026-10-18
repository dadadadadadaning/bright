"""CPU-only orchestration contracts; no BRIGHT data, GPU or real training."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, command='check'):
        return run.parser().parse_args([command, '--output', str(self.root / 'result')])

    def silent(self):
        return redirect_stdout(io.StringIO())

    def stage_stub(self, *, fail=None, incomplete=None, wrong_smoke=None, wrong_provenance=False):
        calls = []
        def child(stage, command, log, state, output):
            calls.append(stage)
            state['current_stage'] = stage
            row = {'status': 'running', 'command': command, 'returncode': 0}
            state['stages'][stage] = row
            log.write_text('synthetic stage only\n', encoding='utf-8')
            if stage == fail:
                row['returncode'] = 1
                raise RuntimeError('synthetic child failure: ' + stage)
            if stage != 'preflight':
                result = Path(command[command.index('--output') + 1])
                result.mkdir()
                smoke = stage in ('parent_smoke', 'build_smoke', 'smoke')
                provenance = 'not_comparable_new_Q_source'
                if wrong_provenance:
                    provenance = 'verified_old_E0'
                (result / 'summary.json').write_text(json.dumps({
                    'complete': stage != incomplete,
                    'smoke': not smoke if stage == wrong_smoke else smoke,
                    'full_validation_evidence': stage == 'train',
                    'old_three_arm_baseline': provenance if stage == 'train' else 'not_applicable',
                }))
            return row
        return child, calls

    def test_plan_without_files_cuda_or_release(self):
        with mock.patch.object(run, 'verify_release', side_effect=AssertionError('not read-only')), \
             mock.patch.object(run, 'check_inputs', side_effect=AssertionError('unexpected inputs')), \
             mock.patch('torch.cuda.is_available', side_effect=AssertionError('unexpected CUDA')), self.silent():
            result = run.main(['plan', '--repo', '/does/not/exist'])
        self.assertTrue(result['read_only'])
        self.assertFalse(result['existing_pooling_fault_fixed'])
        self.assertFalse((self.root / 'result').exists())

    def test_parser_requires_values_for_train(self):
        with redirect_stdout(io.StringIO()), mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            run.parser().parse_args(['train', '--output', str(self.root / 'result')])

    def test_parser_formal_topk_fixed_to_64(self):
        with mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            run.parser().parse_args(['full', '--output', str(self.root / 'result'), '--topk', '8'])

    def test_missing_release_rejected(self):
        with mock.patch.object(run, 'HERE', self.root), self.assertRaisesRegex(ValueError, 'Missing'):
            run.verify_release()

    def make_release(self, names=None):
        (self.root / 'entry.py').write_text('x = 1\n')
        files = names or {'entry.py': run.digest(self.root / 'entry.py')}
        (self.root / 'SHA256SUMS.json').write_text(json.dumps(files))

    def test_release_hashes_python_files(self):
        self.make_release()
        with mock.patch.object(run, 'HERE', self.root):
            self.assertEqual(run.verify_release()['files'], 1)
            (self.root / 'entry.py').write_text('x = 2\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run.verify_release()

    def test_release_rejects_unbound_python(self):
        self.make_release()
        (self.root / 'extra.py').write_text('pass\n')
        with mock.patch.object(run, 'HERE', self.root), self.assertRaisesRegex(ValueError, 'Unbound'):
            run.verify_release()

    def test_release_rejects_parent_escape(self):
        self.make_release({'../outside.py': '0'*64})
        with mock.patch.object(run, 'HERE', self.root), self.assertRaisesRegex(ValueError, 'relative'):
            run.verify_release()

    def test_release_rejects_symlink(self):
        self.make_release()
        (self.root / 'copy.py').symlink_to(self.root / 'entry.py')
        files = {p.name: run.digest(p) for p in self.root.glob('*.py')}
        (self.root / 'SHA256SUMS.json').write_text(json.dumps(files))
        with mock.patch.object(run, 'HERE', self.root), self.assertRaisesRegex(ValueError, 'checksum'):
            run.verify_release()

    def test_check_stops_after_parent_child_and_training_smoke(self):
        child, calls = self.stage_stub()
        with mock.patch.object(run, '_child', child), self.silent():
            result = run.pipeline(self.args())
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke', 'smoke'])
        self.assertTrue(result['complete'])
        self.assertTrue(result['smoke'])
        self.assertFalse(result['full_validation_evidence'])
        self.assertFalse(result['training_complete'])
        self.assertFalse((self.root / 'result/parent').exists())
        self.assertFalse((self.root / 'result/train').exists())

    def test_full_runs_fresh_parent_and_child_banks_before_matched_training(self):
        child, calls = self.stage_stub()
        with mock.patch.object(run, '_child', child), self.silent():
            result = run.pipeline(self.args('full'))
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke', 'smoke',
                                 'parent_full', 'build', 'train'])
        self.assertTrue(result['full_validation_evidence'])
        self.assertEqual(result['old_three_arm_baseline'], 'not_comparable_new_Q_source')
        self.assertFalse(result['historical_e0_source_reverified'])
        self.assertFalse((self.root / 'result/baseline_evidence.json').exists())
        train_command = result['stages']['train']['command']
        self.assertEqual(train_command[train_command.index('--bank-root') + 1],
                         str(self.root / 'result/parent'))
        self.assertEqual(train_command[train_command.index('--values') + 1],
                         str(self.root / 'result/values'))
        self.assertNotIn('--baseline-root', train_command)

    def test_full_rejects_false_old_E0_provenance(self):
        child, calls = self.stage_stub(wrong_provenance=True)
        with mock.patch.object(run, '_child', child), self.silent(), \
             self.assertRaisesRegex(RuntimeError, 'old|Q.source|provenance'):
            run.pipeline(self.args('full'))
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke', 'smoke',
                                 'parent_full', 'build', 'train'])
        state = json.loads((self.root / 'result/workflow_state.json').read_text())
        self.assertFalse(state['complete'])

    def test_launch_check_records_new_source_without_old_E0_claim(self):
        args = self.args('full')
        with mock.patch.object(run, 'verify_release', return_value={'passed': True}), \
             mock.patch.object(run, 'check_inputs', return_value={'passed': True}):
            record = run.validate_start(args)
        self.assertEqual(record['old_baseline_status'], 'not_comparable_new_Q_source')
        self.assertFalse((self.root / 'result').exists())

    def test_failed_smoke_stops_before_formal_build(self):
        child, calls = self.stage_stub(fail='smoke')
        with mock.patch.object(run, '_child', child), self.silent(), \
             self.assertRaisesRegex(RuntimeError, 'synthetic'):
            run.pipeline(self.args('full'))
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke', 'smoke'])
        state = json.loads((self.root / 'result/workflow_state.json').read_text())
        self.assertFalse(state['complete'])
        self.assertFalse(state['full_validation_evidence'])
        self.assertEqual(state['stages']['smoke']['status'], 'failed')
        self.assertTrue((self.root / 'result/traceback.txt').is_file())

    def test_zero_exit_with_incomplete_summary_rejected(self):
        child, calls = self.stage_stub(incomplete='build_smoke')
        with mock.patch.object(run, '_child', child), self.silent(), self.assertRaisesRegex(RuntimeError, 'summary'):
            run.pipeline(self.args())
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke'])

    def test_missing_fresh_parent_summary_stops_before_child_build(self):
        child, calls = self.stage_stub(incomplete='parent_smoke')
        with mock.patch.object(run, '_child', child), self.silent(), self.assertRaisesRegex(RuntimeError, 'summary'):
            run.pipeline(self.args('full'))
        self.assertEqual(calls, ['preflight', 'parent_smoke'])
        self.assertFalse((self.root / 'result/values_smoke').exists())

    def test_incomplete_formal_parent_stops_before_values_or_train(self):
        child, calls = self.stage_stub(incomplete='parent_full')
        with mock.patch.object(run, '_child', child), self.silent(), self.assertRaisesRegex(RuntimeError, 'summary'):
            run.pipeline(self.args('full'))
        self.assertEqual(calls, ['preflight', 'parent_smoke', 'build_smoke', 'smoke', 'parent_full'])
        self.assertFalse((self.root / 'result/values').exists())
        self.assertFalse((self.root / 'result/train').exists())

    def test_wrong_smoke_identity_rejected(self):
        child, calls = self.stage_stub(wrong_smoke='smoke')
        with mock.patch.object(run, '_child', child), self.silent(), self.assertRaisesRegex(RuntimeError, 'summary'):
            run.pipeline(self.args())

    def test_existing_output_preserved(self):
        (self.root / 'result').mkdir()
        sentinel = self.root / 'result/sentinel.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            run.pipeline(self.args())
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_report_reads_failed_partial_without_gates(self):
        output = self.root / 'result'
        output.mkdir()
        (output / 'workflow_state.json').write_text(json.dumps({'complete': False, 'error': 'CUDA failure'}))
        with mock.patch.object(run, 'verify_release', side_effect=AssertionError('Report is read only')), self.silent():
            result = run.main(['report', '--root', str(output)])
        self.assertEqual(result['workflow']['error'], 'CUDA failure')
        self.assertIsNone(result['final'])

    def test_report_prints_four_arm_differences_and_no_false_formal(self):
        output = self.root / 'result'
        (output / 'final_val').mkdir(parents=True)
        (output / 'summary.json').write_text(json.dumps({'complete': False, 'smoke': False, 'full_validation_evidence': False}))
        metrics = {name: {'foreground_miou': value, 'iou': [.95, .7, .4, .7]}
                   for name, value in [('a', .6), ('b', .61), ('c', .62), ('d', .64)]}
        (output / 'final_val/result.json').write_text(json.dumps({'metrics': metrics, 'full_validation_evidence': True,
                                                                'final_integrity_pending': False}))
        capture = io.StringIO()
        with redirect_stdout(capture):
            result = run.report(output)
        self.assertFalse(result['formal'])
        self.assertIn('d-b: +3.0000 pp', capture.getvalue())
        self.assertIn('interaction d-c-b+a: +1.0000 pp', capture.getvalue())

    def test_child_retains_log_and_reports_nonzero(self):
        output = self.root / 'result'
        output.mkdir()
        state = {'stages': {}}
        log = output / 'child.log'
        with self.silent(), self.assertRaisesRegex(RuntimeError, 'returncode 7'):
            run._child('synthetic', [sys.executable, '-c', 'print("debug before failure"); raise SystemExit(7)'], log, state, output)
        self.assertIn('debug before failure', log.read_text())
        self.assertEqual(state['stages']['synthetic']['status'], 'failed')

    def test_train_dispatch_uses_new_source_and_matched_budget(self):
        fake_train = SimpleNamespace(run_training=mock.Mock(return_value={'synthetic': True}), run_evaluation=mock.Mock())
        with mock.patch.dict(sys.modules, {'train': fake_train}), \
             mock.patch.object(run, 'verify_release'), mock.patch.object(run, 'check_inputs'):
            result = run.main(['train', '--output', str(self.root / 'result'),
                               '--bank-root', '/fresh-parent', '--protocol', '/fresh-source/protocol.json',
                               '--values', '/values'])
        args = fake_train.run_training.call_args.args[0]
        self.assertEqual(args.steps, 5000)
        self.assertEqual(args.eval_every, 500)
        self.assertEqual(args.bank_root, '/fresh-parent')
        self.assertIsNone(args.baseline_evidence)
        self.assertEqual(args.baseline_policy, 'independent')
        self.assertTrue(result['synthetic'])

    def test_smoke_dispatch_preserves_small_value_permission_and_identity(self):
        fake_train = SimpleNamespace(run_training=mock.Mock(return_value={'synthetic': True}), run_evaluation=mock.Mock())
        with mock.patch.dict(sys.modules, {'train': fake_train}), \
             mock.patch.object(run, 'verify_release'), mock.patch.object(run, 'check_inputs'):
            run.main(['smoke', '--output', str(self.root / 'result'), '--values', '/small-values',
                      '--bank-root', '/fresh-parent-smoke', '--allow-smoke-values'])
        args = fake_train.run_training.call_args.args[0]
        self.assertTrue(args.smoke)
        self.assertEqual(args.mode, 'smoke')
        self.assertTrue(args.allow_smoke_values)
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.eval_every, 20)

    def test_old_baseline_mode_is_not_an_entry_point(self):
        with mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            run.parser().parse_args(['find-baseline'])

    def test_launch_script_syntax(self):
        for name in ('run.sh', 'start_server.sh'):
            result = subprocess.run(['bash', '-n', str(run.HERE / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
