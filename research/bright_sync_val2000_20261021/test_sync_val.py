"""CPU/synthetic contracts only; not a reproduction of the RTX4090 failure."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'bright_sync_val2000_v1.pyz'
sys.path.insert(0, str(ARCHIVE))
sys.path.append(os.environ['BRIGHT_ABLATION_TEST_SOURCE'])
import torch
import sync_val2000 as program
from evaluation_observer import EvaluationObserver, track_evaluation
import evidence
import test_evaluation as eval_fixtures
spec = importlib.util.spec_from_file_location('report_fixtures',
    Path(os.environ['BRIGHT_RESUME_TEST_SOURCE']) / 'test_evidence.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        eval_fixtures.EvaluationContracts.setUpClass()

    def setUp(self):
        self.fixture = eval_fixtures.EvaluationContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.heads.update(r=copy.deepcopy(self.fixture.heads['d']), l=copy.deepcopy(self.fixture.heads['d']))
        self.runtime = self.fixture.runtime()
        self.observer = EvaluationObserver(self.fixture.folder.parent, dict(index=-1))

    def tracking(self):
        f = self.fixture
        return track_evaluation(self.runtime, f.factory, eval_fixtures.evaluation,
                                self.runtime.val_names, f.legacy.state, self.observer)

    def test_all450_original_evaluator_same_metrics_and_no_optimizer_updates(self):
        f = self.fixture
        self.runtime.val_names = [f'val{i}' for i in range(450)]
        state = f.legacy.state
        before_heads = {a: state.state_fingerprint(h.state_dict()) for a,h in f.heads.items()}
        optimizers = {a: torch.optim.AdamW(h.parameters()) for a,h in f.heads.items()}
        before_opt = {a: state.state_fingerprint(o.state_dict()) for a,o in optimizers.items()}
        before_rng = state.state_fingerprint(state.capture_rng())
        reference = f.run_evaluation(self.runtime, primary_endpoint=False, smoke=False)
        f.folder = f.folder.parent / 'observed'
        with mock.patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('update')):
            with self.tracking():
                report = f.run_evaluation(self.runtime, primary_endpoint=False, smoke=False)
        for key in ('metrics','per_event','transitions','head_hashes','journal_sha256'):
            self.assertEqual(report[key], reference[key], key)
        self.assertEqual(report['image_count'], 450)
        self.assertTrue(report['smoke'])  # CPU data can never count as formal evidence.
        self.assertFalse(report['whole_val450_processed'])
        self.assertEqual(before_heads, {a: state.state_fingerprint(h.state_dict()) for a,h in f.heads.items()})
        self.assertEqual(before_opt, {a: state.state_fingerprint(o.state_dict()) for a,o in optimizers.items()})
        self.assertEqual(before_rng, state.state_fingerprint(state.capture_rng()))
        self.assertIsNone(self.observer.first_failure)
        self.assertNotIn('prepare', vars(self.runtime))
        self.assertNotIn('index_observer', sys.modules)

    def failure_case(self, kind):
        f = self.fixture
        error = RuntimeError('synthetic device-side assert in ' + kind)
        original_forward = f.factory.forward
        if kind == 'forward':
            def forward(head, example, arm, support):
                if arm == 'b':
                    raise error
                return original_forward(head, example, arm, support)
            f.factory.forward = forward
        elif kind == 'loss':
            # Not a wrapped stage: cleanup must see the pending exception.
            self.runtime.loss = mock.Mock(side_effect=error)
        elif kind == 'loader':
            def batches(*args, **kwargs):
                raise error
                yield
            self.runtime.batches = batches
        state = f.legacy.state
        with mock.patch.object(state, 'restore_rng', side_effect=AssertionError('poisoned RNG access')) as restore:
            with self.tracking():
                with self.assertRaises(RuntimeError) as caught:
                    f.run_evaluation(self.runtime)
            self.assertEqual(restore.call_count, 0)
            self.assertIs(state.restore_rng, restore)
        self.assertIs(caught.exception, error)
        self.assertIs(self.observer.first_exception, error)
        self.assertTrue(self.observer.rng_restore_skipped)
        self.assertNotIn('prepare', vars(self.runtime))
        failure = json.loads((f.folder.parent / 'evaluation_failure.json').read_text())
        self.assertIn('synthetic device-side assert', failure['traceback'])
        self.assertFalse(failure['tensor_snapshots_taken'])
        if kind == 'forward':
            self.assertEqual(failure['position']['active_arm'], 'b')
            self.assertEqual(failure['position']['stage'], 'forward')

    def test_forward_error_not_masked_by_cuda_rng_cleanup(self):
        self.failure_case('forward')

    def test_unwrapped_loss_error_not_masked_by_cuda_rng_cleanup(self):
        self.failure_case('loss')

    def test_loader_error_not_masked_by_cuda_rng_cleanup(self):
        self.failure_case('loader')

    def test_failure_evidence_io_error_does_not_mask_original(self):
        with mock.patch('evaluation_observer.write_json', side_effect=OSError('disk full')):
            self.observer.record(ValueError('original'))
        self.assertEqual(str(self.observer.first_exception), 'original')
        self.assertEqual(self.observer.evidence_error, 'disk full')


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / 'sync_val2000_synthetic'
        self.names = [f'val_{i}' for i in range(450)]
        self.original = dict(execution_settings=dict(cuda_launch_blocking=None, precision='fp32'))
        self.actual = copy.deepcopy(self.original)
        self.actual['execution_settings']['cuda_launch_blocking'] = '1'
        self.report = fixtures.make_report(2000, 'val', self.names)
        self.path, self.report = fixtures.write_report(self.output / 'val_step_002000', self.report, self.names)
        self.source = self.base / 'failed_source'
        self.source.mkdir()
        (self.source / 'summary.json').write_text('{"phase":"failed","complete":false}\n')
        self.source_bytes = (self.source / 'summary.json').read_bytes()
        self.checks = dict.fromkeys(program.REQUIRED_CHECKS, True)
        self.provenance = dict(source_root=str(self.source), checkpoint_step=2000,
            protocol_info=copy.deepcopy(self.actual), original_protocol_info=self.original,
            launch_blocking_override=program.validate_protocol(self.actual, self.original))
        program.write_json(self.output / 'manifest.json', self.provenance)

    def finalize(self):
        return program.finalize_fresh_report(self.path, self.report, output=self.output,
            names=self.names, evidence=evidence, checks=self.checks, provenance=self.provenance)

    def test_only_new_complete450_report_finalized_with_explicit_protocol(self):
        before = self.path.read_bytes()
        finished = self.finalize()
        self.assertTrue(finished['complete'])
        self.assertFalse(finished['source_training_run_complete'])
        self.assertFalse(finished['historical_execution_replayed'])
        self.assertEqual(finished['optimizer_updates'], 0)
        self.assertEqual(finished['cuda_launch_blocking'], '1')
        self.assertNotIn('protocol_info', json.loads(before))
        self.assertEqual(finished['protocol_info'], self.actual)
        self.assertEqual(self.path.with_name('result.pending.json').read_bytes(), before)
        self.assertEqual((self.source / 'summary.json').read_bytes(), self.source_bytes)
        evidence.validate_saved_report(self.path, finished, self.names, program.sha256, self.output)

    def test_each_integrity_gate_required(self):
        before = self.path.read_bytes()
        for key in self.checks:
            self.checks[key] = False
            with self.assertRaisesRegex(ValueError, 'integrity checks'):
                self.finalize()
            self.checks[key] = True
        self.assertEqual(self.path.read_bytes(), before)

    def test_partial_or_wrong_step_or_test_or_smoke_rejected(self):
        for key,value in (('image_count',31),('step',1500),('test_accessed',True),('smoke',True)):
            previous = self.report[key]
            self.report[key] = value
            fixtures.rehash_report(self.path, self.report)
            with self.assertRaises(ValueError):
                self.finalize()
            self.report[key] = previous
        self.assertFalse(self.path.with_name('result.pending.json').exists())

    def test_other_execution_change_and_forged_protocol_rejected(self):
        self.provenance['protocol_info']['execution_settings']['precision'] = 'fp16'
        with self.assertRaisesRegex(ValueError, 'beyond declared'):
            self.finalize()
        self.provenance['protocol_info'] = copy.deepcopy(self.actual)
        self.report['protocol_info'] = self.original
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaisesRegex(ValueError, 'execution protocol differs'):
            self.finalize()

    def test_wrong_journal_order_even_if_rehashed_rejected(self):
        journal = self.path.with_name('images.jsonl')
        lines = journal.read_text().splitlines(keepends=True)
        lines[0], lines[1] = lines[1], lines[0]
        journal.write_text(''.join(lines))
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaises(ValueError):
            self.finalize()

    def test_manifest_protocol_mismatch_rejected(self):
        manifest = copy.deepcopy(self.provenance)
        manifest['protocol_info'] = self.original
        program.write_json(self.output / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'execution protocol differs'):
            self.finalize()

    def test_changed_objective_even_if_rehashed_rejected(self):
        journal = self.path.with_name('images.jsonl')
        lines = journal.read_text().splitlines(keepends=True)
        row = json.loads(lines[0])
        row['objectives']['d']['nll'] += .01
        lines[0] = json.dumps(row) + '\n'
        journal.write_text(''.join(lines))
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaises(ValueError):
            self.finalize()

    def test_source_output_or_missing_checksum_rejected(self):
        self.provenance['source_root'] = str(self.base)
        with self.assertRaisesRegex(ValueError, 'outside its source'):
            self.finalize()
        self.provenance['source_root'] = str(self.source)
        self.path.with_name('result.json.sha256').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.finalize()


class LaunchTests(unittest.TestCase):
    def test_worker_environment_and_pinned_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with (base / 'lock').open('a+') as stream:
                with mock.patch.object(program, 'BASE', base), \
                     mock.patch.object(program, 'leases', return_value=contextlib.nullcontext([stream])), \
                     mock.patch.object(program, 'prepared', return_value=dict(evidence_files={})), \
                     mock.patch.object(sys, 'argv', [str(ARCHIVE)]), \
                     mock.patch.object(program.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=123)) as popen:
                    self.assertEqual(program.launch(), 0)
                command = popen.call_args.args[0]
                self.assertEqual(popen.call_args.kwargs['env']['CUDA_LAUNCH_BLOCKING'], '1')
                pinned = Path(command[3])
                self.assertEqual(pinned.read_bytes(), ARCHIVE.read_bytes())
                self.assertEqual(command[4], 'worker')
                self.assertTrue(pinned.parent.name.startswith('sync_val2000_'))
                self.assertEqual(json.loads((pinned.parent / 'summary.json').read_text())['optimizer_updates'], 0)

    def test_inherited_helpers_byte_identical(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            ledger = json.loads(archive.read('SYNC_VAL_BASE.json'))
            for name,digest in ledger['files'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest, name)


if __name__ == '__main__':
    unittest.main()
