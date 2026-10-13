import json
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import diagnose


class DiagnosticContracts(unittest.TestCase):
    def fixture(self, root):
        old = Path(root) / 'old'
        pool = old / 'train/pool_debug'
        pool.mkdir(parents=True)
        snapshot = pool / 'input_000735.pth'
        snapshot.write_bytes(b'fixture only; mocked workers never unpickle this')
        diagnose.write(old / 'train/manifest.json', {
            'protocol_info': {'source_runtime_settings': {'cublas_workspace_config': None,
                                                        'deterministic_algorithms': False}}})
        diagnose.write(pool / 'failure.json', {'input': '/old/server/input_000735.pth', 'image': 'event_0001'})
        tree = ast.parse((diagnose.HERE / 'pool_ops.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_original_algorithm')
        fn.name = 'soft_pool_unpool'
        source = old / 'source_pool.py'
        source.write_text(ast.unparse(fn))
        diagnose.write(pool / 'manifest.json', {'original_source': str(source), 'original_sha256': diagnose.digest(source)})
        return old

    def test_inspection_binds_original_files_without_editing(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            before = {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()}
            result = diagnose.inspect_failed(old)
            self.assertEqual(result['failed_image'], 'event_0001')
            self.assertTrue(result['input'].endswith('input_000735.pth'))
            self.assertEqual(before, {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()})
            for p, h in result['source_files_sha256'].items():
                self.assertEqual(diagnose.digest(p), h)

    def test_missing_snapshot_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            (old / 'train/pool_debug/input_000735.pth').unlink()
            with self.assertRaisesRegex(ValueError, 'exactly the one'):
                diagnose.inspect_failed(old)

    def test_original_source_and_arithmetic_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            evidence = diagnose.inspect_failed(old)
            diagnose.verify_native_source(evidence)
            self.assertTrue(evidence['native_arithmetic_matches_server_ast'])
            source = old / 'source_pool.py'
            source.write_text(source.read_text().replace('1e-06', '0.01'))
            with self.assertRaisesRegex(ValueError, 'differs from the failed'):
                diagnose.verify_native_source(evidence)

    def test_different_source_formula_is_rejected_even_if_hash_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            source = old / 'source_pool.py'
            source.write_text(source.read_text().replace('1e-06', '0.01'))
            diagnose.write(old / 'train/pool_debug/manifest.json', {'original_source': str(source), 'original_sha256': diagnose.digest(source)})
            with self.assertRaisesRegex(ValueError, 'arithmetic differs'):
                diagnose.verify_native_source(diagnose.inspect_failed(old))

    def test_source_runtime_must_not_be_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            diagnose.write(old / 'train/manifest.json', {'protocol_info': {}})
            with self.assertRaisesRegex(ValueError, 'will not guess'):
                diagnose.inspect_failed(old)

    def test_environment_preserves_legacy_without_forcing_determinism(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            env = diagnose.child_environment({'cublas_workspace_config': None})
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', env)
            self.assertEqual(env['CUDA_LAUNCH_BLOCKING'], '1')
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(diagnose.child_environment({'cublas_workspace_config': ':16:8'})['CUBLAS_WORKSPACE_CONFIG'], ':16:8')

    def args(self, folder, old):
        return SimpleNamespace(command='all', failed_run=str(old), tool=str(Path(folder)/'old_tool'),
            repo=folder, output=str(Path(folder)/'new'), device='cuda:0', repeats=3, no_sequence=False)

    def test_native_failure_does_not_prevent_other_isolated_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            commands = []
            def worker(command, destination, env):
                commands.append(command)
                passed = destination.name != 'replay_native'
                return {'passed': passed, 'returncode': 0 if passed else 1, 'summary': {'gates_passed': passed}}
            with patch.object(diagnose, 'verify_release', return_value='release'), patch.object(diagnose, 'run_child', side_effect=worker):
                result = diagnose.run(self.args(folder, old))
            self.assertEqual(len(commands), 5)
            self.assertTrue(result['diagnostic_complete'])
            self.assertFalse(result['all_executed_stages_passed'])
            self.assertFalse(result['training'])
            self.assertFalse(result['full_validation_evidence'])
            self.assertTrue(result['stages']['sequence_val450']['passed'])
            self.assertTrue(all('train' not in c for c in commands))

    def test_invalid_cpu_snapshot_stops_gpu_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            with patch.object(diagnose, 'verify_release', return_value='release'), \
                 patch.object(diagnose, 'run_child', return_value={'passed': False, 'returncode': 1}) as worker:
                with self.assertRaisesRegex(RuntimeError, 'CPU snapshot'):
                    diagnose.run(self.args(folder, old))
            self.assertEqual(worker.call_count, 1)
            self.assertFalse(diagnose.read(Path(folder)/'new/summary.json')['diagnostic_complete'])

    def test_failed_single_image_prevents_redundant_val_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            def worker(command, destination, env):
                passed = destination.name != 'sequence_failed_image'
                return {'passed': passed, 'returncode': 0 if passed else 1}
            with patch.object(diagnose, 'verify_release', return_value='release'), patch.object(diagnose, 'run_child', side_effect=worker):
                result = diagnose.run(self.args(folder, old))
            self.assertTrue(result['stages']['sequence_val450']['skipped'])
            self.assertFalse(result['root_cause_confirmed'])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            old = self.fixture(folder)
            args = self.args(folder, old)
            Path(args.output).mkdir()
            sentinel = Path(args.output) / 'keep'
            sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'NEW output'):
                diagnose.run(args)
            self.assertEqual(sentinel.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
