"""CPU synthetic orchestration contracts; no BRIGHT data, torch or CUDA jobs."""
from __future__ import annotations

import argparse
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
from unittest.mock import patch

import pipeline


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + '\n', encoding='utf-8')


def option(command, name):
    return command[command.index(name) + 1]


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.package = self.repo / Path(*pipeline.MODULE.split('.'))
        self.package.mkdir(parents=True)
        (self.package / 'run.sh').write_text('# synthetic preflight\n')
        (self.package / 'train_q.py').write_text('# synthetic train worker; never executed\n')
        (self.package / 'evaluate_q.py').write_text('# synthetic eval worker; never executed\n')
        self.tool = self.root / 'tool'
        self.tool.mkdir()
        for name in ('pipeline.py', 'verify_sar.py', 'audit_predictions.py', 'gpu_queue.py'):
            (self.tool / name).write_text('# synthetic bound external source; never executed\n')
        self.protocol_path = self.root / 'e0' / 'protocol.json'
        self.inputs = {}
        for name in ('front', 'a1', 'train_list', 'val_list'):
            path = self.root / 'inputs' / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('synthetic input: ' + name + '\n')
            self.inputs[name] = str(path)
        self.protocol = dict(format='SRAC_HARDQUERY_PROTOCOL_V1', scope='all', complete=True,
            smoke=False, image_count=450, full_validation_evidence=True,
            reproduction={'passed': True}, test_accessed=False, profile='legacy', pool_mode='native',
            arguments={'device': 'cuda:0'}, runtime_settings={'cublas_workspace_config': None},
            input_paths=self.inputs,
            input_fingerprints={key: pipeline.digest(path) for key, path in self.inputs.items()})
        write(self.protocol_path, self.protocol)
        self.output = self.root / 'checkpoints' / 'new_run'
        self.args = argparse.Namespace(action='run', repo_root=str(self.repo),
            protocol=str(self.protocol_path), output=str(self.output), python=sys.executable,
            workers=4, amp='bf16', wait_gpu_idle=True)
        self.patch_here = patch.object(pipeline, 'HERE', self.tool)
        self.patch_here.start()
        self.addCleanup(self.patch_here.stop)
        self.called = []

    def queue_module(self):
        return SimpleNamespace(wait_for_idle=lambda device, log: dict(device=device, idle=True))

    def emit_stage(self, stage):
        kind = stage['kind']
        if kind == 'preflight':
            return
        output = Path(stage['output'])
        if kind in ('train', 'evaluate'):
            smoke = stage['smoke']
            steps = 2 if smoke else 2000
            data = dict(complete=True, smoke=smoke, arm=stage['arm'], test_accessed=False,
                        immutable={'weights': True, 'sources': True},
                        full_validation_evidence=not smoke)
            if kind == 'train':
                data.update(format='SRAC_Q_FACTORIAL_V1', steps=steps,
                            a1_unchanged=True, downstream_unchanged=True)
                output.mkdir(parents=True, exist_ok=True)
                (output / 'last.pth').write_bytes(b'SYNTHETIC CHECKPOINT: not torch data')
            else:
                data.update(format='SRAC_Q_FACTORIAL_EVAL_V1',
                            image_count=2 if smoke else 450, checkpoint_steps=steps,
                            native_a1_replay_ok=True, reproduction={'passed': True},
                            q_a1_reproduction={'passed': True}, metrics={
                                'q_region': {'valid_miou': .6144, 'iou': [.96, .75, .43, .66]},
                                'q_grid_probq': {'valid_miou': .6183, 'iou': [.96, .75, .44, .66]}})
            write(output / 'summary.json', data)
        elif kind == 'verify':
            write(output, dict(format='BRIGHT_EXTERNAL_SAR_VALIDATION_V1', complete=True,
                               reports_only=False, test_accessed=False, training=False,
                               direct_differences_pp={'sar_minus_baseline': .1}))
        else:
            write(output / 'report.json', dict(complete=True, test_accessed=False))

    def child(self, stage, repo, env, log):
        self.called.append(stage['name'])
        self.assertEqual(repo, self.repo)
        self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '1')
        self.assertEqual(env['PYTHON'], self.args.python)
        Path(log).write_text('synthetic subprocess log\n')
        self.emit_stage(stage)
        return 0

    def execute(self, child=None, queue=None):
        with patch.dict(sys.modules, {'gpu_queue': queue or self.queue_module()}), \
                patch.object(pipeline, 'run_child', side_effect=child or self.child), \
                redirect_stdout(io.StringIO()):
            return pipeline.run(self.args)

    def test_four_arms_share_parameters_and_smoke_precedes_formal(self):
        before = set(self.root.rglob('*'))
        plan = pipeline.build_plan(self.args)
        self.assertEqual(len(plan), 21)
        self.assertEqual(plan[0]['kind'], 'preflight')
        trains = [s for s in plan if s['kind'] == 'train']
        self.assertEqual([s['arm'] for s in trains], list(pipeline.ARMS) * 2)
        self.assertEqual([s['smoke'] for s in trains], [True] * 4 + [False] * 4)
        for stage in trains:
            command = stage['command']
            budget = '2' if stage['smoke'] else '2000'
            expected = {'--protocol': str(self.protocol_path), '--arm': stage['arm'],
                        '--steps': budget, '--eval_every': budget, '--batch': '2', '--accum': '8',
                        '--workers': '4', '--lr': '0.0001', '--aux_weight': '0.1',
                        '--cue_gain': '0.1', '--amp': 'bf16'}
            for name, value in expected.items():
                self.assertEqual(option(command, name), value)
            self.assertEqual('--smoke' in command, stage['smoke'])
            self.assertFalse(any('test_list' in str(x) for x in command))
        self.assertTrue(all(s.get('smoke') is True for s in plan[1:9]))
        self.assertTrue(all(s.get('smoke') is False for s in plan[9:17]))
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_independent_formal_val_uses_last_checkpoint_without_limit(self):
        plan = pipeline.build_plan(self.args)
        for stage in (s for s in plan if s['kind'] == 'evaluate'):
            command = stage['command']
            scope = 'smoke' if stage['smoke'] else 'formal'
            self.assertEqual(option(command, '--checkpoint'),
                str(self.output / scope / ('train_' + stage['arm']) / 'last.pth'))
            self.assertEqual(option(command, '--protocol'), str(self.protocol_path))
            if stage['smoke']:
                self.assertEqual(option(command, '--limit'), '2')
            else:
                self.assertNotIn('--limit', command)
        self.assertEqual(plan[17]['kind'], 'verify')
        self.assertEqual(plan[18]['kind'], 'paired')
        self.assertEqual([s['kind'] for s in plan[19:]], ['audit', 'audit'])

    def test_all_pass_records_bound_inputs_and_formal_completion(self):
        before = pipeline.snapshot_sources(self.args)
        result = self.execute()
        self.assertEqual(self.called, [s['name'] for s in pipeline.build_plan(self.args)])
        state = pipeline.read_json(self.output / 'workflow_state.json')
        manifest = pipeline.read_json(self.output / 'manifest.json')
        self.assertTrue(state['complete'])
        self.assertTrue(state['full_validation_evidence'])
        self.assertTrue(result['complete'])
        self.assertFalse(result['test_accessed'])
        self.assertEqual(len(state['stages']), 21)
        self.assertTrue(all(s['complete'] and s['gate']['passed'] for s in state['stages']))
        self.assertEqual(manifest['source_bindings'], before)
        self.assertEqual(manifest['protocol_sha256'], pipeline.digest(self.protocol_path))
        self.assertEqual(before, pipeline.snapshot_sources(self.args))
        self.assertAlmostEqual(result['metrics_percentage']['sar_cue']['q_region']['foreground_miou'], 61.44)

    def test_first_smoke_failure_stops_before_any_formal_job(self):
        def fail(stage, repo, env, log):
            value = self.child(stage, repo, env, log)
            return 9 if stage['name'] == 'smoke_train_baseline' else value
        with self.assertRaisesRegex(RuntimeError, 'later stages stopped'):
            self.execute(fail)
        self.assertEqual(self.called, ['preflight', 'smoke_train_baseline'])
        self.assertFalse((self.output / 'formal').exists())
        state = pipeline.read_json(self.output / 'workflow_state.json')
        summary = pipeline.read_json(self.output / 'summary.json')
        self.assertFalse(state['complete'])
        self.assertFalse(state['full_validation_evidence'])
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['full_validation_evidence'])
        self.assertTrue((self.output / 'failure_traceback.txt').is_file())

    def test_formal_returncode_failure_does_not_run_remaining_arms(self):
        def fail(stage, repo, env, log):
            value = self.child(stage, repo, env, log)
            return 1 if stage['name'] == 'formal_train_rgb_capacity' else value
        with self.assertRaisesRegex(RuntimeError, 'formal_train_rgb_capacity'):
            self.execute(fail)
        self.assertEqual(self.called[-1], 'formal_train_rgb_capacity')
        self.assertNotIn('formal_train_shifted_sar', self.called)
        self.assertNotIn('verify_four_arm_artifacts', self.called)
        self.assertFalse(pipeline.summarize(self.output)['full_validation_evidence'])

    def test_exit_zero_with_partial_formal_val_still_stops(self):
        def partial(stage, repo, env, log):
            value = self.child(stage, repo, env, log)
            if stage['name'] == 'formal_val_baseline':
                path = Path(stage['output']) / 'summary.json'
                data = pipeline.read_json(path)
                data['image_count'] = 449
                write(path, data)
            return value
        with self.assertRaisesRegex(ValueError, 'evaluation gate failed'):
            self.execute(partial)
        self.assertEqual(self.called[-1], 'formal_val_baseline')
        self.assertNotIn('formal_train_rgb_capacity', self.called)
        state = pipeline.read_json(self.output / 'workflow_state.json')
        self.assertFalse(state['stages'][-1]['complete'])
        self.assertFalse(state['full_validation_evidence'])

    def test_stage_gates_reject_wrong_budget_missing_last_and_bad_reproduction(self):
        plan = pipeline.build_plan(self.args)
        train = next(s for s in plan if s['name'] == 'formal_train_baseline')
        self.emit_stage(train)
        path = Path(train['output']) / 'summary.json'
        data = pipeline.read_json(path)
        for changes in ({'steps': 1999}, {'complete': False}, {'smoke': True},
                        {'full_validation_evidence': False}, {'immutable': {'code': False}}):
            with self.subTest(changes=changes):
                write(path, dict(data, **changes))
                with self.assertRaises(ValueError):
                    pipeline.validate_stage(train)
        write(path, data)
        (Path(train['output']) / 'last.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'last.pth missing'):
            pipeline.validate_stage(train)
        val = next(s for s in plan if s['name'] == 'formal_val_baseline')
        self.emit_stage(val)
        path = Path(val['output']) / 'summary.json'
        data = pipeline.read_json(path)
        data['reproduction']['passed'] = False
        write(path, data)
        with self.assertRaisesRegex(ValueError, 'reproduction gate failed'):
            pipeline.validate_stage(val)

    def test_invalid_protocol_rejected_before_creating_results(self):
        for changes in ({'complete': False}, {'scope': 'limited'}, {'image_count': 2},
                        {'pool_mode': 'contiguous'}, {'test_accessed': True}):
            with self.subTest(changes=changes):
                write(self.protocol_path, dict(self.protocol, **changes))
                with self.assertRaises(ValueError):
                    self.execute()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.called, [])
        write(self.protocol_path, self.protocol)

    def test_input_hash_mismatch_rejected_before_creating_results(self):
        Path(self.inputs['train_list']).write_text('changed training split\n')
        with self.assertRaisesRegex(ValueError, 'input fingerprint differs'):
            self.execute()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.called, [])

    def test_queued_source_or_protocol_change_stops_before_preflight(self):
        def mutate(device, log):
            (self.package / 'train_q.py').write_text('# modified while waiting\n')
            return dict(idle=True)
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.execute(queue=SimpleNamespace(wait_for_idle=mutate))
        self.assertEqual(self.called, [])
        self.assertFalse(pipeline.summarize(self.output)['complete'])

    def test_final_input_change_cannot_be_certified_complete(self):
        def mutate(stage, repo, env, log):
            value = self.child(stage, repo, env, log)
            if stage['name'] == 'q_audit_sar_cue':
                Path(self.inputs['front']).write_text('changed checkpoint\n')
            return value
        with self.assertRaises(ValueError):
            self.execute(mutate)
        self.assertEqual(self.called[-1], 'q_audit_sar_cue')
        state = pipeline.read_json(self.output / 'workflow_state.json')
        self.assertFalse(state['complete'])
        self.assertFalse(state['full_validation_evidence'])

    def test_report_partial_is_read_only_and_does_not_launch_jobs(self):
        self.output.mkdir(parents=True)
        write(self.output / 'workflow_state.json', dict(complete=False, full_validation_evidence=False,
              current_stage='formal_val_baseline', failure='synthetic interruption'))
        stage = next(s for s in pipeline.build_plan(self.args) if s['name'] == 'formal_val_baseline')
        self.emit_stage(stage)
        before = {str(p): pipeline.digest(p) for p in self.output.rglob('*') if p.is_file()}
        with patch.object(pipeline.subprocess, 'Popen') as process:
            report = pipeline.summarize(self.output)
        process.assert_not_called()
        self.assertFalse(report['complete'])
        self.assertFalse(report['full_validation_evidence'])
        self.assertEqual(len(report['missing_or_unreadable']), 3)
        self.assertIn('baseline', report['metrics_percentage'])
        self.assertEqual(before, {str(p): pipeline.digest(p) for p in self.output.rglob('*') if p.is_file()})

    def test_python_symlink_entrypoint_is_preserved(self):
        interpreter = self.root / 'venv' / 'bin' / 'python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        self.args.python = str(interpreter)
        self.execute()
        manifest = pipeline.read_json(self.output / 'manifest.json')
        self.assertEqual(manifest['arguments']['python'], str(interpreter))
        for stage in manifest['stages']:
            if stage['kind'] != 'preflight':
                self.assertEqual(stage['command'][0], str(interpreter))

    def test_historical_cublas_unset_overrides_contaminated_shell(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            self.execute()
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')
        manifest = pipeline.read_json(self.output / 'manifest.json')
        self.assertIsNone(manifest['runtime_environment']['CUBLAS_WORKSPACE_CONFIG'])

    def test_historical_cublas_value_is_restored(self):
        self.protocol['runtime_settings']['cublas_workspace_config'] = ':4096:8'
        write(self.protocol_path, self.protocol)
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            self.execute()
        manifest = pipeline.read_json(self.output / 'manifest.json')
        self.assertEqual(manifest['runtime_environment']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_queued_source_deletion_stops_before_preflight(self):
        def mutate(device, log):
            (self.package / 'train_q.py').unlink()
            return dict(idle=True)
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.execute(queue=SimpleNamespace(wait_for_idle=mutate))
        self.assertEqual(self.called, [])
        self.assertFalse(pipeline.summarize(self.output)['full_validation_evidence'])

    def test_queued_protocol_rewrite_stops_before_preflight(self):
        def mutate(device, log):
            self.protocol['runtime_settings']['cublas_workspace_config'] = ':4096:8'
            write(self.protocol_path, self.protocol)
            return dict(idle=True)
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.execute(queue=SimpleNamespace(wait_for_idle=mutate))
        self.assertEqual(self.called, [])
        self.assertFalse(pipeline.summarize(self.output)['full_validation_evidence'])

    def test_cli_plan_and_report_do_not_import_torch_or_gpu_queue(self):
        # A fresh interpreter forbids those imports, independent of unittest state.
        real_module_path = Path(pipeline.__file__).resolve()
        script = '''import importlib.abc, importlib.util, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'gpu_queue'):
            raise AssertionError('Unexpected model/GPU import: ' + fullname)
sys.meta_path.insert(0, Forbid())
spec = importlib.util.spec_from_file_location('tested_pipeline', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.argv = ['pipeline.py', sys.argv[2], '--root', sys.argv[3]]
m.main()
'''
        for action in ('plan', 'report'):
            with self.subTest(action=action):
                result = subprocess.run([sys.executable, '-c', script, str(real_module_path), action,
                                         str(self.output)], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                if action == 'plan':
                    self.assertTrue(data['no_training_started'])
                else:
                    self.assertFalse(data['complete'])
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
