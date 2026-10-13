"""CPU orchestration contracts; no BRIGHT tensors, Torch import or GPU work."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('sar_soft_test_pipeline', 'run.py')
worker = load('sar_soft_test_worker', 'worker.py')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


def filesnapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in Path(root).rglob('*') if p.is_file()}


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.source_base = self.root / 'sources'

    def source(self, name, **state):
        root = self.source_base / name
        write(root / 'manifest.json', {'format': runner.SOURCE_FORMAT})
        write(root / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
            complete=False, full_validation_evidence=False, **state))
        return root

    def args(self, output='result'):
        return argparse.Namespace(action='run', repo_root=str(self.repo),
            output=str(self.root / output), sar_root=None,
            source_base=str(self.source_base), python=sys.executable)

    def test_latest_declared_failed_source_is_not_replaced_by_older_success(self):
        older = self.source('run_20260101')
        write(older / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
            complete=True, full_validation_evidence=True))
        latest = self.source('run_20260201', failure='fit failed')
        selected = runner.select_source(base=self.source_base)
        self.assertEqual(selected, latest)
        with self.assertRaisesRegex(ValueError, 'fit failed'):
            runner.wait_source(selected)

    def test_explicit_source_and_unknown_format_selection(self):
        chosen = self.source('run_20260101')
        write(self.source_base / 'run_20990101/manifest.json', {'format': 'OTHER'})
        self.assertEqual(runner.select_source(base=self.source_base), chosen)
        self.assertEqual(runner.select_source(chosen, self.source_base), chosen)
        with self.assertRaisesRegex(ValueError, '不存在'):
            runner.select_source(self.root / 'missing')

    def test_pending_waits_and_accepts_only_formal_completion(self):
        source = self.source('run_wait', current_stage='formal_train_sar')
        log = Mock()
        def complete(_):
            write(source / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
                complete=True, full_validation_evidence=True))
        with patch.object(runner.time, 'sleep', side_effect=complete) as sleep:
            runner.wait_source(source, log=log, poll_seconds=0)
        sleep.assert_called_once_with(0)
        self.assertIn('formal_train_sar', log.call_args.args[0])
        write(source / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
            complete=True, full_validation_evidence=False))
        with self.assertRaisesRegex(ValueError, 'formal evidence'):
            runner.wait_source(source)

    def test_waiting_source_failure_stops_instead_of_polling_forever(self):
        source = self.source('run_wait')
        def fail(_):
            write(source / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
                complete=False, failure='CUDA failure'))
        with patch.object(runner.time, 'sleep', side_effect=fail) as sleep:
            with self.assertRaisesRegex(ValueError, 'CUDA failure'):
                runner.wait_source(source, log=Mock(), poll_seconds=0)
        self.assertEqual(sleep.call_count, 1)

    def test_wait_keeps_locked_source_when_a_newer_run_appears(self):
        selected = self.source('run_20260101')
        locked = runner.select_source(base=self.source_base)
        def finish_locked(_):
            self.source('run_20990101', failure='new unrelated run failed')
            write(selected / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
                complete=True, full_validation_evidence=True))
        with patch.object(runner.time, 'sleep', side_effect=finish_locked):
            runner.wait_source(locked, log=Mock(), poll_seconds=0)
        self.assertEqual(locked, selected)
        self.assertNotEqual(runner.select_source(base=self.source_base), locked)

    def test_plan_has_all_smoke_before_any_formal_and_exact_action_arguments(self):
        args = self.args()
        args.sar_root = str(self.root / 'source')
        stages = runner.plan(args)
        self.assertEqual(stages[0]['kind'], 'certificate')
        self.assertEqual(stages[1]['kind'], 'preflight')
        work = [s for s in stages if s['kind'] in ('train', 'evaluate')]
        self.assertEqual(len(work), 8)
        self.assertEqual([s['smoke'] for s in work], [True] * 4 + [False] * 4)
        self.assertEqual([(s['arm'], s['kind']) for s in work[:4]],
            [('baseline', 'train'), ('baseline', 'evaluate'), ('sar_cue', 'train'), ('sar_cue', 'evaluate')])
        for stage in work:
            command = stage['command']
            self.assertEqual(command[3], stage['kind'])
            self.assertEqual('--smoke' in command, stage['smoke'])
            self.assertEqual('--fit' in command, stage['kind'] == 'evaluate')
            if stage['kind'] == 'evaluate':
                self.assertTrue(command[command.index('--fit') + 1].endswith('fit_' + stage['arm']))
        self.assertEqual(stages[-1]['kind'], 'compare')

    def stage(self, kind='train', smoke=False):
        output = self.root / ('stage_' + kind)
        stage = dict(name='synthetic_' + kind, kind=kind, output=str(output), smoke=smoke, arm='baseline')
        data = dict(format='SRAC_SAR_SOFT_ROUTE_V1' + ('_EVAL' if kind == 'evaluate' else ''),
                    complete=True, smoke=smoke, full_validation_evidence=not smoke,
                    test_accessed=False)
        if kind == 'train':
            steps = 2 if smoke else 2000
            data.update(steps=steps, updates_per_arm={arm: steps for arm in runner.ROUTES})
        else:
            data['image_count'] = 2 if smoke else 450
        record = dict(format='SAR_SOFT_WORKER_V1', complete=True, q_arm='baseline', action=kind, smoke=smoke)
        def commit():
            write(output / 'summary.json', data)
            record['summary_sha256'] = runner.digest(output / 'summary.json')
            write(output / 'worker_record.json', record)
        commit()
        return stage, data, record, commit

    def test_stage_gate_rejects_incomplete_budget_and_wrong_smoke_scope(self):
        stage, data, record, commit = self.stage()
        self.assertTrue(runner.stage_gate(stage)['passed'])
        data['updates_per_arm']['soft'] = 1999
        commit()
        with self.assertRaisesRegex(ValueError, 'fixed update budget'):
            runner.stage_gate(stage)
        data['updates_per_arm']['soft'] = 2000
        data['smoke'] = True
        commit()
        with self.assertRaisesRegex(ValueError, 'gate failed'):
            runner.stage_gate(stage)

    def test_stage_gate_rejects_worker_arm_or_checksum_and_eval_count(self):
        stage, data, record, commit = self.stage('evaluate')
        self.assertTrue(runner.stage_gate(stage)['passed'])
        record['q_arm'] = 'sar_cue'
        commit()
        with self.assertRaisesRegex(ValueError, 'gate failed'):
            runner.stage_gate(stage)
        record['q_arm'] = 'baseline'
        data['image_count'] = 449
        commit()
        with self.assertRaisesRegex(ValueError, 'image count'):
            runner.stage_gate(stage)
        data['image_count'] = 450
        commit()
        record['summary_sha256'] = 'invalid'
        write(Path(stage['output']) / 'worker_record.json', record)
        with self.assertRaisesRegex(ValueError, 'gate failed'):
            runner.stage_gate(stage)

    def test_stage_gate_rejects_wrong_summary_and_worker_formats(self):
        for kind in ('train', 'evaluate'):
            stage, data, record, commit = self.stage(kind)
            expected = data['format']
            data['format'] = 'SRAC_BUILDING_ROUTE_V1'
            commit()
            with self.assertRaisesRegex(ValueError, 'gate failed'):
                runner.stage_gate(stage)
            data['format'] = expected
            record['format'] = 'OTHER_WORKER'
            commit()
            with self.assertRaisesRegex(ValueError, 'gate failed'):
                runner.stage_gate(stage)

    def test_certificate_and_comparison_gates_require_exact_formats(self):
        for kind, expected in (('certificate', 'SAR_SOFT_SOURCE_CERTIFICATE_V1'),
                               ('compare', 'SRAC_SAR_SOFT_CROSS_COMPARISON_V1')):
            output = self.root / kind
            file = output if kind == 'certificate' else output / 'report.json'
            stage = dict(kind=kind, output=str(output))
            data = dict(format=expected, complete=True, full_validation_evidence=True, test_accessed=False)
            write(file, data)
            self.assertTrue(runner.stage_gate(stage)['passed'])
            data['format'] = 'OTHER_FORMAL_REPORT'
            write(file, data)
            with self.assertRaises(ValueError):
                runner.stage_gate(stage)

    def test_report_reads_failed_pipeline_without_writing_or_resuming(self):
        root = self.root / 'failed'
        write(root / 'workflow_state.json', dict(complete=False, full_validation_evidence=False,
            current_stage='formal_train_baseline', failure='RuntimeError: synthetic failure'))
        (root / 'comparison').mkdir()
        (root / 'comparison/report.json').write_text('incomplete-json')
        before = filesnapshot(root)
        report = runner.report(root)
        self.assertFalse(report['complete'])
        self.assertIn('synthetic failure', report['failure'])
        self.assertIn('comparison_read_error', report)
        self.assertEqual(filesnapshot(root), before)

    def test_fresh_parent_import_does_not_import_torch_or_allocate_cuda(self):
        code = (
            'import importlib.util,sys; '
            's=importlib.util.spec_from_file_location("standalone",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'assert "torch" not in sys.modules; '
            'assert not any(n.startswith("torch.") for n in sys.modules)'
        )
        result = subprocess.run([sys.executable, '-B', '-c', code, str(HERE / 'run.py')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_failure(self, mode):
        source = self.source('run_20260301')
        write(source / 'workflow_state.json', dict(format=runner.SOURCE_FORMAT,
            complete=True, full_validation_evidence=True))
        protocol = self.root / 'protocol.json'
        write(protocol, {'arguments': {'device': 'cuda:0'}})
        calls = []
        def child(stage, repo, env, log):
            calls.append(stage['name'])
            Path(log).write_text('synthetic child log\n')
            if stage['kind'] == 'certificate':
                write(stage['output'], dict(format='SAR_SOFT_SOURCE_CERTIFICATE_V1',
                    complete=True, full_validation_evidence=True,
                    protocol=str(protocol), source_runtime_settings={'cublas_workspace_config': None}))
                return 0
            if stage['kind'] == 'preflight':
                return 7 if mode == 'exit' else 0
            self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '1')
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', env)
            write(Path(stage['output']) / 'summary.json', dict(format='SRAC_SAR_SOFT_ROUTE_V1', complete=False))
            write(Path(stage['output']) / 'worker_record.json', dict(format='SAR_SOFT_WORKER_V1', complete=False))
            return 0
        queue = ModuleType('gpu_queue')
        queue.wait_for_idle = Mock(return_value={'synthetic_idle': True})
        with patch.object(runner, 'source_files', return_value={'fixture': 'sha'}), \
             patch.object(runner, 'child', side_effect=child), \
             patch.dict(sys.modules, {'gpu_queue': queue}), contextlib.redirect_stdout(None):
            with self.assertRaises((RuntimeError, ValueError)):
                runner.run(self.args())
        state = runner.read(self.root / 'result/workflow_state.json')
        self.assertFalse(state['complete'])
        self.assertFalse(state['full_validation_evidence'])
        self.assertTrue(state['failure'])
        self.assertEqual(state['source_root'], str(source))
        self.assertTrue((self.root / 'result/summary.json').is_file())
        self.assertFalse(any(name.startswith('formal_') for name in calls))
        return calls, queue

    def test_nonzero_child_stops_before_gpu_work(self):
        calls, queue = self.run_failure('exit')
        self.assertEqual(calls, ['verify_sar_source', 'original_route_preflight'])
        queue.wait_for_idle.assert_not_called()

    def test_zero_exit_with_failed_worker_gate_stops_all_later_stages(self):
        calls, queue = self.run_failure('gate')
        self.assertEqual(calls[-1], 'smoke_train_baseline')
        self.assertEqual(len(calls), 3)
        queue.wait_for_idle.assert_called_once()


class WorkerContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.protocol = self.root / 'e0.json'
        write(self.protocol, {'fixture': 'E0, no tensors'})
        self.q = {}
        for arm in ('baseline', 'sar_cue'):
            p = self.root / (arm + '.pth')
            p.write_bytes(('not-a-real-checkpoint-' + arm).encode())
            self.q[arm] = p
        self.certificate = self.root / 'certificate.json'
        self.cert = dict(format='SAR_SOFT_SOURCE_CERTIFICATE_V1', complete=True,
            full_validation_evidence=True, repo_root=str(self.repo), protocol=str(self.protocol),
            protocol_sha256=worker.digest(self.protocol), baseline_checkpoint=str(self.q['baseline']),
            sar_checkpoint=str(self.q['sar_cue']), source_runtime_settings={'cublas_workspace_config': None},
            source_bindings={str(p): worker.digest(p) for p in [self.protocol, *self.q.values()]})
        write(self.certificate, self.cert)
        self.env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1')
        self.env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        self.old_path = list(sys.path)
        self.old_bytecode = sys.dont_write_bytecode
        self.addCleanup(lambda: setattr(sys, 'path', self.old_path))
        self.addCleanup(lambda: setattr(sys, 'dont_write_bytecode', self.old_bytecode))
        self.calls = []
        self.effect = None
        self.result = {'complete': True}
        self.adapter = ModuleType('core_adapter')
        self.adapter.precheck_fit = Mock()
        self.adapter.source_hash = Mock(return_value='synthetic-adapter-sha')
        modules = {action: SimpleNamespace(parser=lambda a=action: self.action_parser(a),
                    run=lambda args, a=action: self.execute(a, args)) for action in ('train', 'evaluate')}
        @contextlib.contextmanager
        def adapted():
            yield modules
        self.adapter.adapted_modules = adapted

    def action_parser(self, action):
        parser = argparse.ArgumentParser()
        for name in ('protocol', 'output'):
            parser.add_argument('--' + name, required=True)
        if action == 'train':
            parser.add_argument('--q-checkpoint', required=True)
            for name in ('steps', 'eval-every'):
                parser.add_argument('--' + name, type=int, required=True)
            for name in ('lr', 'threshold', 'soft-floor'):
                parser.add_argument('--' + name, type=float, required=True)
            parser.add_argument('--smoke', action='store_true')
        else:
            parser.add_argument('--fit', required=True)
            parser.add_argument('--limit', type=int, default=None)
        return parser

    def execute(self, action, args):
        self.calls.append((action, args))
        write(Path(args.output) / 'summary.json', {
            'format': 'SRAC_SAR_SOFT_ROUTE_V1' + ('_EVAL' if action == 'evaluate' else ''),
            'complete': True, 'synthetic': True})
        if action == 'evaluate':
            write(Path(args.output) / 'result.json', {'complete': True, 'synthetic': True})
        if self.effect:
            self.effect(action, args)
        return self.result

    def args(self, action='train', smoke=False, arm='baseline', output='worker_output'):
        return argparse.Namespace(repo_root=str(self.repo), certificate=str(self.certificate),
            q_arm=arm, output=str(self.root / output), smoke=smoke, action=action, fit=None)

    def invoke(self, args):
        with patch.dict(os.environ, self.env, clear=True), \
             patch.dict(sys.modules, {'core_adapter': self.adapter}), contextlib.redirect_stdout(None):
            return worker.run(args)

    def make_fit(self, arm='baseline'):
        root = self.root / 'fit'
        write(root / 'summary.json', {'q_source': {'arm': arm, 'sha256': worker.digest(self.q[arm])}})
        write(root / 'manifest.json', {'fixture': True})
        (root / 'heads_last.pth').write_bytes(b'fixture')
        (root / 'training.jsonl').write_text('{}\n')
        return root

    def test_train_action_has_fixed_smoke_and_formal_budgets(self):
        for smoke in (True, False):
            args = self.args(smoke=smoke, arm='sar_cue', output=str(smoke))
            result = self.invoke(args)
            action, parsed = self.calls[-1]
            self.assertEqual(action, 'train')
            self.assertEqual(parsed.q_checkpoint, str(self.q['sar_cue']))
            self.assertEqual(parsed.steps, 2 if smoke else 2000)
            self.assertEqual(parsed.eval_every, parsed.steps)
            self.assertEqual((parsed.lr, parsed.threshold, parsed.soft_floor), (1e-4, .5, 0.0))
            self.assertEqual(parsed.smoke, smoke)
            self.assertTrue(result['complete'])
            self.assertEqual(result['summary_sha256'], worker.digest(Path(args.output) / 'summary.json'))
        self.adapter.precheck_fit.assert_not_called()

    def test_evaluation_uses_fit_precheck_and_limit_only_for_smoke(self):
        fit = self.make_fit()
        before = filesnapshot(fit)
        for smoke in (True, False):
            args = self.args(action='evaluate', smoke=smoke, output=str(smoke))
            args.fit = str(fit)
            record = self.invoke(args)
            self.adapter.precheck_fit.assert_called_with(str(fit), str(self.protocol), smoke=smoke)
            action, parsed = self.calls[-1]
            self.assertEqual(action, 'evaluate')
            self.assertEqual(parsed.limit, 2 if smoke else None)
            self.assertEqual(record['fit_checkpoint_sha256'], worker.digest(fit / 'heads_last.pth'))
            self.assertEqual(record['result_sha256'], worker.digest(Path(args.output) / 'result.json'))
        self.assertEqual(filesnapshot(fit), before)

    def test_sync_environment_and_cublas_match_are_required_before_adapter(self):
        self.env['CUDA_LAUNCH_BLOCKING'] = '0'
        with self.assertRaisesRegex(ValueError, 'CUDA_LAUNCH_BLOCKING'):
            self.invoke(self.args())
        self.env['CUDA_LAUNCH_BLOCKING'] = '1'
        self.env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        with self.assertRaisesRegex(ValueError, 'CUBLAS'):
            self.invoke(self.args())
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'worker_output').exists())

    def test_certificate_repository_and_source_integrity_are_required(self):
        self.cert['repo_root'] = str(self.root / 'wrong_repo')
        write(self.certificate, self.cert)
        with self.assertRaisesRegex(ValueError, 'certificate'):
            self.invoke(self.args())
        self.cert['repo_root'] = str(self.repo)
        write(self.certificate, self.cert)
        self.q['baseline'].write_bytes(b'changed-source')
        with self.assertRaisesRegex(ValueError, 'Certified input changed'):
            self.invoke(self.args())
        self.assertEqual(self.calls, [])

    def test_evaluation_rejects_fit_from_other_q_arm_before_loading(self):
        args = self.args(action='evaluate', arm='sar_cue')
        args.fit = str(self.make_fit('baseline'))
        with self.assertRaisesRegex(ValueError, 'selected certified Q'):
            self.invoke(args)
        self.assertEqual(self.calls, [])

    def test_evaluation_precheck_failure_does_not_execute_gpu_module(self):
        args = self.args(action='evaluate')
        args.fit = str(self.make_fit())
        self.adapter.precheck_fit.side_effect = ValueError('fit protocol mismatch')
        with self.assertRaisesRegex(ValueError, 'fit protocol mismatch'):
            self.invoke(args)
        self.assertEqual(self.calls, [])

    def test_post_execution_input_mutation_marks_worker_failed(self):
        self.effect = lambda action, args: self.protocol.write_text('changed during worker')
        args = self.args()
        with self.assertRaisesRegex(ValueError, 'Certified input changed while'):
            self.invoke(args)
        record = worker.read(Path(args.output) / 'worker_record.json')
        self.assertFalse(record['complete'])
        self.assertIn('Certified input changed', record['error'])

    def test_post_evaluation_fit_mutation_marks_worker_failed(self):
        fit = self.make_fit()
        self.effect = lambda action, args: (fit / 'heads_last.pth').write_bytes(b'changed training weights')
        args = self.args(action='evaluate')
        args.fit = str(fit)
        with self.assertRaisesRegex(ValueError, 'Training evidence changed'):
            self.invoke(args)
        self.assertFalse(worker.read(Path(args.output) / 'worker_record.json')['complete'])

    def test_source_hash_change_marks_worker_failed(self):
        self.adapter.source_hash.side_effect = ['before', 'after']
        args = self.args()
        with self.assertRaisesRegex(ValueError, 'Experiment source changed'):
            self.invoke(args)
        self.assertFalse(worker.read(Path(args.output) / 'worker_record.json')['complete'])

    def test_incomplete_result_does_not_become_complete_worker_evidence(self):
        self.result = {'complete': False, 'reason': 'synthetic incomplete run'}
        args = self.args()
        with self.assertRaisesRegex(ValueError, 'incomplete evidence'):
            self.invoke(args)
        record = worker.read(Path(args.output) / 'worker_record.json')
        self.assertFalse(record['complete'])
        self.assertIn('incomplete evidence', record['error'])

    def test_worker_refuses_existing_output_and_output_under_repository(self):
        args = self.args()
        Path(args.output).mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.invoke(args)
        args.output = str(self.repo / 'new_result')
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            self.invoke(args)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
