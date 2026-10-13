"""CPU-only launcher contracts: synthetic evidence, no GPU or BRIGHT data."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import start_joint_sync as launcher


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


class LauncherContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tool = self.root / 'New' / 'original_joint_tool'
        self.tool.mkdir(parents=True)
        (self.tool / 'run.py').write_text('# synthetic launcher source; never executed\n')
        (self.tool / 'common.py').write_text('# synthetic common source\n')
        (self.tool / 'test_original.py').write_text('# synthetic original test source\n')
        hashes = {p.name: launcher.digest(p) for p in self.tool.glob('*.py')}
        dump(self.tool / 'SHA256SUMS.json', hashes)
        source = hashlib.sha256()
        for p in sorted(self.tool.glob('*.py')):
            if not p.name.startswith('test_'):
                source.update(p.name.encode())
                source.update(p.read_bytes())
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.base = self.root / 'new_results'
        self.diag_root = self.root / 'preserved_diagnosis'
        self.manifest_path = self.root / 'old_result' / 'train' / 'manifest.json'
        self.values = dict(mode='train', steps=5000, repo=str(self.repo),
                           v3_root=str(self.root / 'paired_v3'), device='cuda:0',
                           workers=4, seed=3407, topk=64, chunk_size=64,
                           lr=0.0001, weight_decay=0.005, protocol=None,
                           output=str(self.manifest_path.parent))
        self.binding = dict(source_protocol=str(self.root / 'e0' / 'protocol.json'),
                            source_runtime_settings=dict(cublas_workspace_config=None,
                                cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True),
                            paired_bank_sha256='synthetic_paired_bank')
        self.manifest = dict(arguments=self.values, protocol_info=self.binding,
                             source_hash=source.hexdigest())
        dump(self.manifest_path, self.manifest)
        self.seq_path = self.diag_root / 'sequence_val450' / 'summary.json'
        self.seq = dict(diagnostic_complete=True, gates_passed=True, scope='val450',
                        whole_val450_processed=True, image_count=450, training=False,
                        pool_guard_report=dict(backend='native', restored=True,
                                               failures=[], restore_conflicts=[]),
                        heads_unchanged=True,
                        initial_probability_checks={a: dict(passed=True) for a in ('r0', 'rt', 'rf')},
                        environment=dict(CUDA_LAUNCH_BLOCKING='1'),
                        preserved_manifest=str(self.manifest_path),
                        preserved_manifest_sha256=launcher.digest(self.manifest_path),
                        protocol_info=self.binding, arguments=dict(tool=str(self.tool)),
                        original_release=dict(package_sha256=launcher.digest(self.tool / 'SHA256SUMS.json')))
        dump(self.seq_path, self.seq)
        self.diag_path = self.diag_root / 'summary.json'
        self.diag = dict(diagnostic_complete=True, all_executed_stages_passed=True,
                         stages={s: dict(passed=True, returncode=0, skipped=False) for s in (
                             'replay_cpu', 'replay_native', 'replay_contiguous',
                             'sequence_failed_image', 'sequence_val450')})
        dump(self.diag_path, self.diag)
        self.args = argparse.Namespace(diag_root=str(self.diag_root), tool=str(self.tool),
                                       result_base=str(self.base), dry_run=False)

    def prepare(self):
        return launcher.prepare(self.diag_root, self.tool, self.base)

    def input_fingerprints(self):
        return {str(p): launcher.digest(p) for directory in (self.tool, self.diag_root,
                                                             self.manifest_path.parent)
                for p in directory.rglob('*') if p.is_file()}

    def test_passed_evidence_pins_original_full_command_and_environment(self):
        before = self.input_fingerprints()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8',
                                     'JOINT_SEED': '111', 'JOINT_TOPK': '8'}):
            record, env, cwd = self.prepare()
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')
        command = [launcher.sys.executable, '-u', str(self.tool / 'run.py'), 'full']
        for name in ('repo', 'v3_root', 'device', 'workers', 'seed', 'topk', 'chunk_size', 'lr', 'weight_decay'):
            command.extend(['--' + name.replace('_', '-'), str(self.values[name])])
        command.extend(['--protocol', self.binding['source_protocol'], '--output', record['output']])
        self.assertEqual(record['command'], command)
        self.assertEqual(cwd, self.repo)
        self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '1')
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', env)
        self.assertEqual(record['pool_backend'], 'native')
        self.assertEqual(record['steps_per_arm'], 5000)
        self.assertEqual(record['smoke_steps_per_arm'], 20)
        self.assertFalse(record['training_completed'])
        self.assertFalse(record['full_validation_evidence'])
        self.assertFalse(record['root_cause_claim'])
        self.assertFalse(self.base.exists())
        self.assertEqual(before, self.input_fingerprints())

    def test_original_explicit_protocol_and_cublas_are_retained(self):
        self.manifest['arguments']['protocol'] = self.binding['source_protocol']
        self.binding['source_runtime_settings']['cublas_workspace_config'] = ':4096:8'
        dump(self.manifest_path, self.manifest)
        self.seq['preserved_manifest_sha256'] = launcher.digest(self.manifest_path)
        dump(self.seq_path, self.seq)
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            record, env, _ = self.prepare()
        self.assertEqual(record['command'].count('--protocol'), 1)
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_skipped_or_failed_diagnostic_stage_rejects_before_launch(self):
        for stage in tuple(self.diag['stages']):
            for change in ({'skipped': True}, {'passed': False}, {'returncode': 1}):
                with self.subTest(stage=stage, change=change):
                    row = self.diag['stages'][stage]
                    original = row.copy()
                    row.update(change)
                    dump(self.diag_path, self.diag)
                    with patch.object(launcher.subprocess, 'Popen') as popen:
                        with self.assertRaises(ValueError):
                            launcher.launch(self.args)
                        popen.assert_not_called()
                    self.assertFalse(self.base.exists())
                    row.clear()
                    row.update(original)
        dump(self.diag_path, self.diag)

    def test_incomplete_val_or_head_gates_reject_before_launch(self):
        cases = [dict(diagnostic_complete=False), dict(gates_passed=False),
                 dict(scope='failed_image'), dict(whole_val450_processed=False),
                 dict(image_count=449), dict(training=True), dict(heads_unchanged=False),
                 dict(environment={'CUDA_LAUNCH_BLOCKING': None}),
                 dict(initial_probability_checks={a: dict(passed=(a != 'rf'))
                                                   for a in ('r0', 'rt', 'rf')}),
                 dict(pool_guard_report=dict(backend='contiguous', restored=True,
                                             failures=[], restore_conflicts=[])),
                 dict(pool_guard_report=dict(backend='native', restored=False,
                                             failures=[], restore_conflicts=[]))]
        for change in cases:
            with self.subTest(change=change):
                dump(self.seq_path, dict(self.seq, **change))
                with patch.object(launcher.subprocess, 'Popen') as popen:
                    with self.assertRaises(ValueError):
                        launcher.launch(self.args)
                    popen.assert_not_called()
                self.assertFalse(self.base.exists())
        dump(self.seq_path, self.seq)

    def test_changed_original_tool_rejected_even_with_updated_local_manifest(self):
        (self.tool / 'run.py').write_text('# changed code\n')
        with self.assertRaisesRegex(ValueError, 'Original tool changed'):
            self.prepare()
        hashes = launcher.read(self.tool / 'SHA256SUMS.json')
        hashes['run.py'] = launcher.digest(self.tool / 'run.py')
        dump(self.tool / 'SHA256SUMS.json', hashes)
        with self.assertRaisesRegex(ValueError, 'source hash changed'):
            self.prepare()

    def test_extra_python_file_is_not_allowed(self):
        (self.tool / 'injected.py').write_text('# unbound file\n')
        with self.assertRaisesRegex(ValueError, 'Unbound Python file'):
            self.prepare()

    def test_old_manifest_change_rejected(self):
        self.manifest['arguments']['seed'] = 123
        dump(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.prepare()

    def test_protocol_binding_mismatch_rejected(self):
        self.seq['protocol_info'] = dict(self.binding, paired_bank_sha256='another_bank')
        dump(self.seq_path, self.seq)
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            self.prepare()

    def test_nonformal_original_budget_rejected(self):
        self.manifest['arguments']['steps'] = 20
        dump(self.manifest_path, self.manifest)
        self.seq['preserved_manifest_sha256'] = launcher.digest(self.manifest_path)
        dump(self.seq_path, self.seq)
        with self.assertRaisesRegex(ValueError, '5000-step'):
            self.prepare()

    def test_dry_run_writes_nothing_and_launches_nothing(self):
        self.args.dry_run = True
        before = self.input_fingerprints()
        with patch.object(launcher.subprocess, 'Popen') as popen, redirect_stdout(io.StringIO()) as output:
            record = launcher.launch(self.args)
        popen.assert_not_called()
        self.assertIn('Dry run only', output.getvalue())
        self.assertFalse(record['launched'])
        self.assertFalse(self.base.exists())
        self.assertEqual(before, self.input_fingerprints())

    def test_mocked_launch_records_fresh_pid_log_and_preserves_old_files(self):
        before = self.input_fingerprints()
        with patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(pid=123456)) as popen, redirect_stdout(io.StringIO()):
            record = launcher.launch(self.args)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        kwargs = popen.call_args.kwargs
        self.assertEqual(command, record['command'])
        self.assertEqual(command[3], 'full')
        self.assertEqual(kwargs['env']['CUDA_LAUNCH_BLOCKING'], '1')
        self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', kwargs['env'])
        self.assertEqual(kwargs['cwd'], self.repo)
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['stdin'], launcher.subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], launcher.subprocess.STDOUT)
        result = Path(record['output'])
        self.assertEqual(result.parent, self.base)
        self.assertFalse(result.exists(), 'The original run.py must create its fresh output itself')
        log = result.with_suffix('.log')
        self.assertEqual(kwargs['stdout'].name, str(log))
        self.assertTrue(log.is_file())
        saved = launcher.read(result.with_suffix('.launch.json'))
        self.assertTrue(saved['launched'])
        self.assertEqual(saved['pid'], 123456)
        self.assertEqual(saved['log'], str(log))
        self.assertFalse(saved['training_completed'])
        self.assertEqual(saved['old_manifest_sha256'], launcher.digest(self.manifest_path))
        self.assertEqual(before, self.input_fingerprints())

    def test_launch_failure_is_recorded_without_claiming_success(self):
        with patch.object(launcher.subprocess, 'Popen', side_effect=OSError('synthetic launch failure')):
            with self.assertRaisesRegex(OSError, 'synthetic launch failure'):
                launcher.launch(self.args)
        notes = list(self.base.glob('*.launch.json'))
        self.assertEqual(len(notes), 1)
        saved = launcher.read(notes[0])
        self.assertFalse(saved['launched'])
        self.assertIn('synthetic launch failure', saved['launch_error'])
        self.assertNotIn('pid', saved)

    def test_new_results_cannot_be_placed_inside_preserved_tool_or_diagnosis(self):
        for base in (self.tool / 'new', self.diag_root / 'new'):
            with self.subTest(base=base):
                with self.assertRaisesRegex(ValueError, 'outside preserved'):
                    launcher.prepare(self.diag_root, self.tool, base)
                self.assertFalse(base.exists())


if __name__ == '__main__':
    unittest.main()
