"""Coordinator contracts using fake workers; no torch, CUDA, or training."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import audit


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


class BindingAndOrchestration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'recovery'
        self.debug = self.root / 'train/pool_debug'
        self.debug.mkdir(parents=True)
        self.snapshot = self.debug / 'input_000001.pth'
        self.snapshot.write_bytes(b'CPU fixture; never deserialized')
        self.failure = dict(phase='cpu_reference_gate', input=str(self.snapshot), image='event_0001', call=1)
        self.runtime = dict(cublas_workspace_config=None, environment=dict(CUDA_VISIBLE_DEVICES='GPU-saved',
                            OMP_NUM_THREADS='2', MKL_NUM_THREADS=None, NVIDIA_TF32_OVERRIDE=None))
        self.evidence = dict(runtime=copy.deepcopy(self.runtime), arguments=dict(device='cuda:0'),
                             failed_run='/preserved/original', checkpoint_step=1500,
                             protocol_info=dict(source_runtime_settings=copy.deepcopy(self.runtime)))
        dump(self.debug / 'failure.json', self.failure)
        dump(self.root / 'runtime.json', self.runtime)
        dump(self.root / 'evidence.json', self.evidence)
        self.q, self.soft = self.base / 'q', self.base / 'soft'
        dump(self.q / 'workflow_state.json', dict(complete=False, failure='source failed',
             current_stage='formal', stages=[dict(name='fit', status='failed', returncode=1, log='logs/fit.log')]))
        (self.q / 'logs').mkdir()
        (self.q / 'logs/fit.log').write_text('preserved traceback\n', encoding='utf-8')
        dump(self.soft / 'workflow_state.json', dict(complete=False, failure=None, current_stage='waiting_sar_source'))
        self.events = []
        self.numeric_bad = set()
        self.execution_bad = set()
        self.bad_identity = set()

    def tearDown(self):
        self.temp.cleanup()

    def args(self, cpu_only=False):
        parent = self.base / 'numeric'
        return SimpleNamespace(recovery_root=str(self.root), output_parent=str(parent), output=str(parent / 'new'),
                               repeats=3, cpu_only=cpu_only, sar_roots=[str(self.q), str(self.soft)], report=None)

    def fake_child(self, command, log, env):
        def option(name):
            return command[command.index('--' + name) + 1]
        backend = option('backend')
        self.events.append(backend)
        self.assertNotIn('train', command)
        output = Path(option('output'))
        output.mkdir()
        Path(log).write_text('fake ' + backend + '\n', encoding='utf-8')
        bad = backend in self.execution_bad
        record = dict(execution_complete=not bad, finite_outputs=not bad, failure={'message': 'CUDA assert'} if bad else None,
                      image=option('expected-image'), call=int(option('expected-call')),
                      input_sha256=option('expected-sha256'), comparisons={'reference': {'passed': backend not in self.numeric_bad}})
        if backend in self.bad_identity:
            record['image'] = 'foreign'
        if backend == 'compare':
            root = Path(option('audit-root'))
            missing = [name for name in ('cpu', 'native', 'contiguous')
                       if not (root / name / 'summary.json').is_file() or name in self.execution_bad]
            record.update(diagnostic_complete=not missing, missing_workers=missing)
        dump(output / 'summary.json', record)
        return 1 if bad else 0

    def fake_idle(self, device, **kwargs):
        self.events.append('idle')
        self.assertEqual(device, 0)
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), 'GPU-saved')
        return dict(idle=True, exclusive_reservation=False)

    def invoke(self, args=None):
        args = args or self.args()
        queue = SimpleNamespace(wait_for_idle=self.fake_idle)
        with mock.patch.object(audit, 'verify_release', return_value='release-sha'), \
             mock.patch.object(audit, 'run_child', side_effect=self.fake_child), \
             mock.patch.dict(sys.modules, {'gpu_queue': queue}), redirect_stdout(io.StringIO()):
            return audit.run(args)

    def test_binding_uses_new_numeric_snapshot_and_exact_runtime(self):
        value = audit.inspect_recovery(self.root)
        self.assertEqual(value['snapshot'], str(self.snapshot))
        self.assertEqual(value['failed_call'], 1)
        self.assertEqual(value['snapshot_sha256'], audit.digest(self.snapshot))

    def test_reject_original_kernel_failure(self):
        self.failure['phase'] = 'pool_kernel'
        dump(self.debug / 'failure.json', self.failure)
        with self.assertRaisesRegex(ValueError, 'cpu_reference_gate'):
            audit.inspect_recovery(self.root)

    def test_reject_snapshot_outside_debug(self):
        outside = self.root / 'outside.pth'
        outside.write_bytes(b'outside')
        self.failure['input'] = str(outside)
        dump(self.debug / 'failure.json', self.failure)
        with self.assertRaisesRegex(ValueError, 'beneath'):
            audit.inspect_recovery(self.root)

    def test_reject_snapshot_symlink_escape(self):
        outside = self.base / 'outside.pth'
        outside.write_bytes(b'outside')
        self.snapshot.unlink()
        self.snapshot.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'beneath'):
            audit.inspect_recovery(self.root)

    def test_reject_runtime_and_protocol_mismatch(self):
        changed = copy.deepcopy(self.runtime)
        changed['cublas_workspace_config'] = ':4096:8'
        dump(self.root / 'runtime.json', changed)
        with self.assertRaisesRegex(ValueError, 'runtime.json'):
            audit.inspect_recovery(self.root)
        dump(self.root / 'runtime.json', self.runtime)
        self.evidence['protocol_info']['source_runtime_settings'] = changed
        dump(self.root / 'evidence.json', self.evidence)
        with self.assertRaisesRegex(ValueError, 'protocol runtime'):
            audit.inspect_recovery(self.root)

    def test_reject_device_and_bad_call(self):
        self.evidence['arguments']['device'] = 'cuda'
        dump(self.root / 'evidence.json', self.evidence)
        with self.assertRaisesRegex(ValueError, 'cuda:N'):
            audit.inspect_recovery(self.root)
        self.evidence['arguments']['device'] = 'cuda:0'
        dump(self.root / 'evidence.json', self.evidence)
        self.failure['call'] = True
        dump(self.debug / 'failure.json', self.failure)
        with self.assertRaisesRegex(ValueError, 'positive call'):
            audit.inspect_recovery(self.root)

    def test_preserved_snapshot_change_rejected(self):
        value = audit.inspect_recovery(self.root)
        self.snapshot.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            audit.check_preserved(value)

    def test_child_environment_restores_source_and_removes_conflicts(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-wrong', 'CUBLAS_WORKSPACE_CONFIG': ':16:8',
                                         'NVIDIA_TF32_OVERRIDE': '0', 'OMP_NUM_THREADS': '9', 'MKL_NUM_THREADS': '9'}):
            env = audit.child_environment(self.runtime)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-saved')
        self.assertEqual(env['OMP_NUM_THREADS'], '2')
        self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '1')
        for key in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'MKL_NUM_THREADS'):
            self.assertNotIn(key, env)

    def test_reject_bad_environment_type(self):
        value = copy.deepcopy(self.runtime)
        value['environment']['CUDA_VISIBLE_DEVICES'] = 0
        with self.assertRaises(TypeError):
            audit.child_environment(value)

    def test_gpu_mapping_scope_restored_even_on_failure(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-old', 'CUDA_DEVICE_ORDER': 'OLD'}):
            with self.assertRaises(RuntimeError):
                with audit.selected_gpu_environment({'CUDA_VISIBLE_DEVICES': 'GPU-new'}):
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-new')
                    self.assertNotIn('CUDA_DEVICE_ORDER', os.environ)
                    raise RuntimeError('fixture')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-old')
            self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'OLD')

    def test_sar_collection_reports_failure_and_waiting_without_liveness_claim(self):
        before = audit.digest(self.q / 'workflow_state.json')
        result = audit.collect_sar_status([self.q, self.soft])
        self.assertEqual(result['runs'][0]['status'], 'failed')
        self.assertEqual(result['runs'][0]['failed_stages'][0]['log_tail'], ['preserved traceback'])
        self.assertEqual(result['runs'][1]['status'], 'waiting_sar_source')
        self.assertFalse(result['process_liveness_checked'])
        self.assertEqual(audit.digest(self.q / 'workflow_state.json'), before)

    def test_sar_failure_log_cannot_escape_run(self):
        outside = self.base / 'outside.log'
        outside.write_text('do not read', encoding='utf-8')
        dump(self.q / 'workflow_state.json', dict(failure='failed', stages={'bad': {'status': 'failed', 'log': str(outside)}}))
        result = audit.collect_sar_status([self.q])
        record = result['runs'][0]['failed_stages'][0]
        self.assertNotIn('log_tail', record)
        self.assertTrue(record['log_errors'])

    def test_cpu_only_never_waits_for_gpu_and_marks_partial(self):
        result = self.invoke(self.args(cpu_only=True))
        self.assertEqual(self.events, ['cpu', 'compare'])
        self.assertTrue(result['execution_complete'])
        self.assertFalse(result['diagnostic_complete'])
        self.assertEqual(result['missing_workers'], ['native', 'contiguous'])
        self.assertFalse(result['training_gate_passed'])
        self.assertFalse(result['training_started'])

    def test_cpu_numeric_disagreement_does_not_prevent_isolated_cuda(self):
        self.numeric_bad.add('cpu')
        result = self.invoke()
        self.assertEqual(self.events, ['cpu', 'idle', 'native', 'contiguous', 'compare'])
        self.assertTrue(result['diagnostic_complete'])
        self.assertFalse(result['all_numeric_passed'])
        self.assertEqual(result['stages']['cpu']['status'], 'numeric_disagreement')

    def test_native_failure_does_not_poison_or_prevent_candidate_worker(self):
        self.execution_bad.add('native')
        result = self.invoke()
        self.assertEqual(self.events, ['cpu', 'idle', 'native', 'contiguous', 'compare'])
        self.assertTrue(result['execution_complete'])
        self.assertFalse(result['diagnostic_complete'])
        self.assertFalse(result['all_workers_executed_successfully'])
        self.assertEqual(result['stages']['native']['status'], 'execution_failed')
        self.assertEqual(result['stages']['contiguous']['status'], 'complete')

    def test_cpu_execution_failure_stops_before_gpu(self):
        self.execution_bad.add('cpu')
        with self.assertRaisesRegex(RuntimeError, 'no CUDA'):
            self.invoke()
        self.assertEqual(self.events, ['cpu'])
        result = audit.read(Path(self.args().output) / 'summary.json')
        self.assertFalse(result['execution_complete'])
        self.assertFalse(result['training_started'])

    def test_foreign_worker_identity_stops_before_gpu(self):
        self.bad_identity.add('cpu')
        with self.assertRaisesRegex(RuntimeError, 'CPU input'):
            self.invoke()
        self.assertEqual(self.events, ['cpu'])
        result = audit.read(Path(self.args().output) / 'summary.json')
        self.assertIn('identity differs', result['stages']['cpu']['error'])

    def test_existing_output_is_never_overwritten(self):
        args = self.args()
        Path(args.output).mkdir(parents=True)
        marker = Path(args.output) / 'preserved'
        marker.write_bytes(b'old')
        with self.assertRaises(FileExistsError):
            self.invoke(args)
        self.assertEqual(marker.read_bytes(), b'old')
        self.assertEqual(self.events, [])

    def test_output_must_be_beneath_declared_parent(self):
        args = self.args()
        args.output = str(self.base / 'elsewhere')
        with self.assertRaisesRegex(ValueError, 'beneath'):
            self.invoke(args)
        self.assertFalse(Path(args.output).exists())

    def test_output_cannot_be_in_preserved_recovery(self):
        args = self.args()
        args.output_parent = str(self.root)
        args.output = str(self.root / 'new')
        with self.assertRaisesRegex(ValueError, 'outside preserved'):
            self.invoke(args)

    def test_recursive_numeric_verdicts_include_cuda_repeat_gates(self):
        self.assertFalse(audit.numeric_passed({'comparisons': {}, 'repeats': [
            {'vs_cpu64': {'passed': True}, 'original_cpu32_gate': {'passed': False}}]}))
        self.assertIsNone(audit.numeric_passed({'comparisons': {}, 'repeats': []}))

    def test_bad_repeats_are_rejected_before_output(self):
        args = self.args()
        args.repeats = 0
        with self.assertRaisesRegex(ValueError, '1..10'):
            self.invoke(args)
        self.assertFalse(Path(args.output).exists())

    def test_report_is_compact_and_explains_partial_cpu_evidence(self):
        result = self.invoke(self.args(cpu_only=True))
        path = Path(self.args().output) / 'report.txt'
        text = path.read_text(encoding='utf-8')
        self.assertIn('CPU-only', text)
        self.assertIn('training_started=false', text)
        self.assertIn('未检查进程存活', text)
        self.assertIn('waiting_sar_source', text)
        self.assertNotIn('worst_failing_examples', text)
        self.assertFalse(result['training_gate_passed'])


class ReleaseAndSubprocess(unittest.TestCase):
    def test_release_manifest_rejects_changed_or_unbound_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'audit.py'
            source.write_text('# release fixture', encoding='utf-8')
            dump(root / 'SHA256SUMS.json', {'audit.py': audit.digest(source)})
            audit.verify_release(root)
            (root / 'extra.py').write_text('# unbound', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unbound'):
                audit.verify_release(root)
            (root / 'extra.py').unlink()
            source.write_text('# changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                audit.verify_release(root)

    def test_child_is_a_new_subprocess_and_log_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'child.log'
            code = audit.run_child([sys.executable, '-c', 'print("isolated child")'], log, os.environ.copy())
            self.assertEqual(code, 0)
            self.assertEqual(audit.tail(log), ['isolated child'])
            with self.assertRaises(FileExistsError):
                audit.run_child([sys.executable, '-c', 'print("second")'], log, os.environ.copy())

    def test_long_worker_emits_heartbeat_without_long_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'long.log'
            child = mock.Mock()
            child.wait.side_effect = [audit.subprocess.TimeoutExpired(['worker'], 30), 0]
            output = io.StringIO()
            with mock.patch.object(audit.subprocess, 'Popen', return_value=child), redirect_stdout(output):
                self.assertEqual(audit.run_child(['fake-worker'], log, os.environ.copy()), 0)
            self.assertIn('RUNNING long.log', output.getvalue())
            self.assertEqual(child.wait.call_args_list, [mock.call(timeout=30), mock.call(timeout=30)])


if __name__ == '__main__':
    unittest.main()
