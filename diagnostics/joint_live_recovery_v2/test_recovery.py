"""Standard-library orchestration tests; fake workers never access CUDA or train."""
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

import recovery


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


class LiveParentContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.original = self.base / 'original'
        self.original.mkdir()
        self.weights = self.original / 'checkpoint.pt'
        self.weights.write_bytes(b'trusted fixture only; never loaded')
        self.repo, self.tool = self.base / 'repo', self.base / 'tool'
        self.repo.mkdir(); self.tool.mkdir()
        runtime = dict(cublas_workspace_config=None, environment={'CUDA_VISIBLE_DEVICES': 'GPU-saved'})
        self.binding = dict(failed_run=str(self.original), tool=str(self.tool), repo=str(self.repo),
            arguments={'mode': 'train', 'steps': 5000, 'device': 'cuda:0'}, checkpoint=str(self.weights),
            checkpoint_step=1500, original_source_hash='original-source',
            protocol_info={'source_runtime_settings': runtime}, runtime=runtime,
            failed_image='old_0001', failed_call=9005,
            preserved_sha256={str(self.weights): recovery.digest(self.weights)})
        self.latest = self.base / 'latest'
        self.debug = self.latest / 'train/pool_debug'
        self.debug.mkdir(parents=True)
        self.snapshot = self.debug / 'input_000002.pth'
        self.snapshot.write_bytes(b'latest numeric input fixture')
        self.failure = dict(phase='cpu_reference_gate', input=str(self.snapshot), image='first_val', call=2)
        dump(self.debug / 'failure.json', self.failure)
        dump(self.latest / 'evidence.json', self.binding)
        dump(self.latest / 'runtime.json', runtime)
        self.latest_summary = dict(phase='checkpoint_revalidation', complete=False, error='CPU reference gate failed',
                                  updates_per_arm={a:1500 for a in ('r0', 'rt', 'rf')},
                                  new_updates_per_arm={a:0 for a in ('r0', 'rt', 'rf')})
        dump(self.latest / 'train/summary.json', self.latest_summary)
        manifest = dict(source_hash='original-source', protocol_info={**self.binding['protocol_info'], 'recovery_binding': {}})
        dump(self.latest / 'train/manifest.json', manifest)
        dump(self.latest / 'workflow_state.json', dict(complete=False, failure='revalidation failed'))
        self.events = []
        self.fail_stage = None
        self.bad_cpu_full = False
        self.bad_cpu_analysis = False
        self.bad_live_control = False
        self.live_numeric_disagreement = True

    def tearDown(self):
        self.temp.cleanup()

    def args(self, stage='recover'):
        parent = self.base / 'results'
        return SimpleNamespace(stage=stage, failed_run=str(self.original), tool=str(self.tool),
            latest_failed_recovery=str(self.latest), output_parent=str(parent), output=str(parent / 'new'),
            after_sar=['q', 'soft'], wait_timeout=1, root=None)

    def fake_original(self, root, tool):
        self.assertEqual(Path(root), self.original)
        self.assertEqual(Path(tool), self.tool)
        return copy.deepcopy(self.binding)

    def fake_child(self, command, log, env):
        def option(key):
            return command[command.index('--' + key) + 1]
        mode = option('mode')
        name = mode if Path(command[2]).name == 'worker.py' else mode + '_analysis'
        self.events.append(name)
        destination = Path(option('output'))
        destination.mkdir()
        Path(log).write_text('fake isolated ' + name + '\nTraceback fixture\n', encoding='utf-8')
        if name == self.fail_stage:
            dump(destination / 'summary.json', {'execution_complete': False, 'failure': {'phase': 'pool_kernel'}})
            return 1
        evidence = recovery.read(option('evidence'))
        if name.endswith('probe'):
            backend = recovery.LIVE_BACKEND if mode == 'live-probe' else recovery.CPU_BACKEND
            capture = destination / 'capture.pth'
            capture.write_bytes(('actual packet ' + name).encode())
            record = dict(execution_complete=True, capture_ready=True, capture=str(capture),
                          capture_sha256=recovery.digest(capture), image=evidence['probe_image'],
                          call=evidence['probe_call'], backend=backend, probe_only=True,
                          new_updates_per_arm={'r0':0,'rt':0,'rf':0}, strict_cpu_gate_passed=True,
                          full_single_image_forward_complete=not self.bad_cpu_full, failure=None)
            if mode == 'live-probe' and self.live_numeric_disagreement:
                record.update(gate_passed=False, strict_cpu_gate_passed=False,
                    failure={'phase':'cpu_reference_gate', 'diagnostic_reference_mismatch':True})
        elif name.endswith('analysis'):
            record = dict(execution_complete=True, diagnostic_complete=True, adequate_capture=True,
                          cpu64_reference_agreement=not (name == 'live_analysis' and self.bad_live_control),
                          cpu32_reference_reproduced=True, CPU_backend_vs_FP64_passed=not self.bad_cpu_analysis,
                          cpu_continuation_checks_passed=not self.bad_cpu_analysis,
                          capture=option('input'), capture_sha256=option('expected-sha256'),
                          evidence_sha256=recovery.digest(option('evidence')),
                          capture_meta={'image':option('expected-image'), 'call':int(option('expected-call')),
                                        'backend':option('expected-backend')}, failure=None)
        else:
            recovery.check_train_gates(option('evidence'))
            record = dict(phase='complete', complete=True, full_validation_evidence=True,
                          updates_per_arm={'r0':5000,'rt':5000,'rf':5000},
                          new_updates_per_arm={'r0':3500,'rt':3500,'rf':3500})
        dump(destination / 'summary.json', record)
        return 0

    def wait_idle(self, index, **kwargs):
        self.events.append('idle')
        self.assertEqual(index, 0)
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), 'GPU-saved')
        return {'idle':True, 'exclusive_reservation':False}

    def invoke(self, args=None):
        queue = SimpleNamespace(wait_for_idle=self.wait_idle)
        with mock.patch.object(recovery, 'verify_release', return_value='release-sha'), \
             mock.patch.object(recovery, 'source_hash', return_value='new-source-sha'), \
             mock.patch.object(recovery, 'inspect_failed', side_effect=self.fake_original), \
             mock.patch.object(recovery, 'wait_sources', side_effect=lambda *a:self.events.append('sar') or ['terminal']), \
             mock.patch.object(recovery, 'terminal_source', return_value=(True,'failed')), \
             mock.patch.object(recovery, 'run_child', side_effect=self.fake_child), \
             mock.patch.dict(sys.modules, {'gpu_queue':queue}), redirect_stdout(io.StringIO()):
            return recovery.pipeline(args or self.args())

    def inspect(self):
        with mock.patch.object(recovery, 'inspect_failed', side_effect=self.fake_original):
            return recovery.inspect_sources(self.original, self.tool, self.latest)

    def test_latest_binding_keeps_original_hash_dictionary_unchanged(self):
        evidence = self.inspect()
        self.assertEqual(evidence['preserved_sha256'], self.binding['preserved_sha256'])
        self.assertEqual(evidence['failed_image'], 'old_0001')
        self.assertEqual(evidence['probe_image'], 'first_val')
        self.assertIn(str(self.snapshot), evidence['diagnostic_source_sha256'])

    def test_latest_failure_must_be_numeric_reference_failure(self):
        self.failure['phase'] = 'pool_kernel'
        dump(self.debug / 'failure.json', self.failure)
        with self.assertRaisesRegex(ValueError, 'cpu_reference_gate'):
            self.inspect()

    def test_latest_snapshot_cannot_escape_debug_through_symlink(self):
        outside = self.latest / 'outside.pth'
        outside.write_bytes(b'outside')
        self.snapshot.unlink(); self.snapshot.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.inspect()

    def test_latest_snapshot_and_json_are_preserved(self):
        evidence = self.inspect()
        self.snapshot.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'numeric-failure evidence changed'):
            recovery.check_preserved(evidence)

    def test_latest_recovery_cannot_contain_new_training_updates(self):
        self.latest_summary['new_updates_per_arm']['rt'] = 1
        dump(self.latest / 'train/summary.json', self.latest_summary)
        with self.assertRaisesRegex(ValueError, 'zero new updates'):
            self.inspect()

    def test_latest_runtime_and_checkpoint_must_match_original(self):
        changed = recovery.read(self.latest / 'evidence.json')
        changed['checkpoint'] = '/another/checkpoint.pt'
        dump(self.latest / 'evidence.json', changed)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.inspect()

    def test_inspect_does_not_launch_or_create_outputs(self):
        result = self.invoke(self.args(stage='inspect'))
        self.assertEqual(result['probe_image'], 'first_val')
        self.assertEqual(self.events, [])
        self.assertFalse(Path(self.args().output).exists())

    def test_live_numeric_disagreement_is_observational_then_explicit_cpu_gate(self):
        state = self.invoke()
        self.assertEqual(self.events, ['sar','idle','live-probe','live_analysis','cpu-probe','cpu_analysis','train'])
        self.assertTrue(state['complete'])
        self.assertTrue(state['full_validation_evidence'])
        self.assertTrue(state['training_started'])
        self.assertFalse(state['automatic_fallback'])
        self.assertFalse(state['tolerance_relaxed'])

    def test_live_kernel_failure_stops_without_cpu_probe_or_training(self):
        self.fail_stage = 'live-probe'
        with self.assertRaisesRegex(RuntimeError, 'live-probe'):
            self.invoke()
        self.assertEqual(self.events, ['sar','idle','live-probe'])
        state = recovery.read(Path(self.args().output) / 'workflow_state.json')
        self.assertFalse(state['training_started'])

    def test_bad_independent_live_reference_stops_before_cpu_probe(self):
        self.bad_live_control = True
        with self.assertRaisesRegex(RuntimeError, 'live_analysis'):
            self.invoke()
        self.assertNotIn('cpu-probe', self.events)
        self.assertNotIn('train', self.events)

    def test_cpu_probe_requires_full_image_and_all_heads_forward(self):
        self.bad_cpu_full = True
        with self.assertRaisesRegex(RuntimeError, 'cpu-probe'):
            self.invoke()
        self.assertNotIn('cpu_analysis', self.events)
        self.assertNotIn('train', self.events)

    def test_cpu_fp64_gate_failure_prevents_training(self):
        self.bad_cpu_analysis = True
        with self.assertRaisesRegex(RuntimeError, 'cpu_analysis'):
            self.invoke()
        self.assertNotIn('train', self.events)
        state = recovery.read(Path(self.args().output) / 'workflow_state.json')
        self.assertFalse(state['probe_gates_passed'])
        self.assertFalse(state['training_started'])

    def test_cpu_aggregate_gate_is_required_even_when_individual_booleans_pass(self):
        record = dict(execution_complete=True, diagnostic_complete=True, adequate_capture=True,
                      cpu64_reference_agreement=True, cpu32_reference_reproduced=True,
                      CPU_backend_vs_FP64_passed=True, cpu_continuation_checks_passed=False, failure=None)
        self.assertFalse(recovery._analysis_ready(record, cpu=True))
        self.assertTrue(recovery._analysis_ready(record, cpu=False))

    def test_probe_only_stops_after_references_without_training(self):
        state = self.invoke(self.args(stage='probe'))
        self.assertNotIn('train', self.events)
        self.assertTrue(state['probe_gates_passed'])
        self.assertFalse(state['training_gate_passed'])
        self.assertFalse(state['training_started'])
        self.assertFalse(state['full_validation_evidence'])

    def test_training_failure_preserves_prefix_and_no_false_completion(self):
        self.fail_stage = 'train'
        with self.assertRaisesRegex(RuntimeError, 'Stage failed: train'):
            self.invoke()
        state = recovery.read(Path(self.args().output) / 'workflow_state.json')
        self.assertFalse(state['complete'])
        self.assertEqual(state['updates_per_arm'], {'r0':1500,'rt':1500,'rf':1500})

    def test_direct_train_gate_rejects_missing_parent_workflow(self):
        path = self.base / 'manual/evidence.json'
        value = self.inspect()
        dump(path, value)
        with self.assertRaises(FileNotFoundError):
            recovery.check_train_gates(path)

    def test_probe_only_artifacts_do_not_authorize_direct_train_worker(self):
        self.invoke(self.args(stage='probe'))
        with self.assertRaisesRegex(ValueError, 'parent recovery workflow'):
            recovery.check_train_gates(Path(self.args().output) / 'evidence.json')

    def test_changed_completed_analysis_cannot_authorize_training(self):
        self.invoke(self.args(stage='probe'))
        root = Path(self.args().output)
        state = recovery.read(root / 'workflow_state.json')
        state['arguments']['stage'] = 'recover'
        state['current_stage'] = 'train'
        state['stages']['train'] = {'status':'running'}
        dump(root / 'workflow_state.json', state)
        analysis = root / 'cpu_analysis/summary.json'
        data = recovery.read(analysis); data['adequate_capture'] = False; dump(analysis, data)
        with mock.patch.object(recovery, 'verify_release', return_value='release-sha'), \
             mock.patch.object(recovery, 'source_hash', return_value='new-source-sha'):
            with self.assertRaisesRegex(ValueError, 'changed persisted preflight'):
                recovery.check_train_gates(root / 'evidence.json')

    def test_analysis_evidence_hash_is_required_not_optional(self):
        self.invoke(self.args(stage='probe'))
        root = Path(self.args().output)
        state = recovery.read(root / 'workflow_state.json')
        state['arguments']['stage'] = 'recover'
        state['current_stage'] = 'train'
        state['stages']['train'] = {'status':'running'}
        path = root / 'cpu_analysis/summary.json'
        data = recovery.read(path); data.pop('evidence_sha256'); dump(path, data)
        state['stages']['cpu_analysis']['summary_sha256'] = recovery.digest(path)
        dump(root / 'workflow_state.json', state)
        with mock.patch.object(recovery, 'verify_release', return_value='release-sha'), \
             mock.patch.object(recovery, 'source_hash', return_value='new-source-sha'):
            with self.assertRaisesRegex(ValueError, 'different parent evidence'):
                recovery.check_train_gates(root / 'evidence.json')

    def test_no_existing_output_overwrite(self):
        args = self.args()
        Path(args.output).mkdir(parents=True)
        marker = Path(args.output) / 'old'
        marker.write_bytes(b'old')
        with self.assertRaises(FileExistsError):
            self.invoke(args)
        self.assertEqual(marker.read_bytes(), b'old')
        self.assertEqual(self.events, [])

    def test_output_must_be_outside_old_source_and_beneath_parent(self):
        args = self.args(); args.output = str(self.base / 'outside')
        with self.assertRaisesRegex(ValueError, 'outside selected'):
            self.invoke(args)
        args.output_parent = str(self.original); args.output = str(self.original / 'new')
        with self.assertRaisesRegex(ValueError, 'outside preserved'):
            self.invoke(args)

    def test_capture_identity_and_checksum_are_required(self):
        root = self.base / 'capture'
        root.mkdir(); path = root / 'capture.pth'; path.write_bytes(b'packet')
        row = {'capture_ready':True, 'capture':str(path), 'capture_sha256':recovery.digest(path),
               'image':'first_val', 'call':3, 'backend':recovery.CPU_BACKEND,
               'probe_only':True, 'new_updates_per_arm':{'r0':0,'rt':0,'rf':0}}
        with self.assertRaisesRegex(ValueError, 'identity/backend'):
            recovery._capture(row, root, 'first_val', 2, recovery.CPU_BACKEND)
        row['call'] = 2; row['capture_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            recovery._capture(row, root, 'first_val', 2, recovery.CPU_BACKEND)

    def test_report_refreshes_train_progress(self):
        self.invoke(self.args(stage='probe'))
        root = Path(self.args().output)
        dump(root / 'train/summary.json', {'phase':'training_continuation', 'updates_per_arm':{'r0':1520,'rt':1520,'rf':1520},
                                         'new_updates_per_arm':{'r0':20,'rt':20,'rf':20}})
        with redirect_stdout(io.StringIO()):
            result = recovery.report(root)
        self.assertTrue(result['training_started'])
        self.assertIn('1520', (root / 'report.txt').read_text())


class ProcessAndLockContracts(unittest.TestCase):
    def test_worker_waits_have_30_second_heartbeat(self):
        child = mock.Mock()
        child.wait.side_effect = [recovery.subprocess.TimeoutExpired(['worker'],30), 0]
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with mock.patch.object(recovery.subprocess, 'Popen', return_value=child), redirect_stdout(output):
                code = recovery.run_child(['fake'], Path(temp)/'child.log', os.environ.copy())
        self.assertEqual(code, 0)
        self.assertIn('RUNNING child.log', output.getvalue())
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=30),mock.call(timeout=30)])

    def test_keyboard_interrupt_terminates_owned_child_before_releasing_lock(self):
        child = mock.Mock()
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(recovery.subprocess, 'Popen', return_value=child):
                with self.assertRaises(KeyboardInterrupt):
                    recovery.run_child(['fake'], Path(temp)/'child.log', os.environ.copy())
        child.terminate.assert_called_once()
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=30),mock.call(timeout=15)])

    def test_new_recovery_and_legacy_share_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            base, old = Path(temp)/'new', Path(temp)/'legacy'
            with recovery.recovery_lock(base, old):
                with self.assertRaisesRegex(RuntimeError, 'Another Memory recovery'):
                    with recovery.source_binding.recovery_lock(old):
                        self.fail('overlap should be rejected')

    def test_gpu_environment_scope_restores_mapping(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-old'}):
            with recovery.gpu_environment({'CUDA_VISIBLE_DEVICES':'GPU-saved'}):
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-saved')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-old')


if __name__ == '__main__':
    unittest.main()
