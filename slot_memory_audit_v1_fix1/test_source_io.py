"""Small file-fixture tests; no BRIGHT data, model deserialization or CUDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import source_io as source

HAS_TORCH = importlib.util.find_spec('torch') is not None


class SourceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root, self.tool = base / 'run', base / 'tool'
        self.train = self.root / 'train'
        self.tool.mkdir()
        (self.train / 'checkpoints').mkdir(parents=True)
        (self.tool / 'common.py').write_text('# static provenance fixture only\n')
        (self.tool / 'test_original.py').write_text('# bound synthetic test\n')
        (self.tool / 'README.md').write_text('synthetic fixture\n')
        self.write(self.tool / 'SHA256SUMS.json', {
            p.name: source.digest(p) for p in self.tool.iterdir() if p.is_file()})
        self.expected = source._source_hash(self.tool)
        patch = mock.patch.object(source, 'EXPECTED_SOURCE', self.expected)
        patch.start()
        self.addCleanup(patch.stop)
        self.protocol = {'execution_settings': {'cuda_launch_blocking': '1'}, 'source': 'fixture'}
        self.config = {'steps': 5000, 'eval_every': 500, 'output': str(self.train),
                       'baseline_policy': 'independent', 'bank_root': str(base / 'bank'),
                       'values': str(base / 'values'), 'protocol': str(base / 'source.json')}
        self.manifest = {'format': source.SOURCE_FORMAT, 'source_hash': self.expected,
                         'protocol_info': self.protocol, 'arguments': self.config,
                         'train_names': ['train_%04d' % i for i in range(2918)],
                         'val_names': ['val_%04d' % i for i in range(450)]}
        self.write(self.train / 'manifest.json', self.manifest)
        complete = {'format': source.SOURCE_FORMAT, 'complete': True,
                    'full_validation_evidence': True, 'smoke': False}
        self.write(self.root / 'workflow_state.json', complete)
        self.write(self.train / 'summary.json', complete)
        for step in (500, 1500, 5000):
            self.make_checkpoint(step)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def make_checkpoint(self, step):
        name = 'checkpoint_last.pt' if step == 5000 else 'checkpoint_%06d.pt' % step
        path = self.train / 'checkpoints' / name
        path.write_bytes(('inert serialized fixture %d' % step).encode())
        meta = dict(format=source.SOURCE_FORMAT, file=name, step=step, mode='train',
                    smoke=False, training_complete=step == 5000,
                    source_hash=self.expected, protocol_info=self.protocol,
                    sha256=source.digest(path))
        self.write(path.with_suffix('.pt.json'), meta)
        report = dict(image_count=450, execution_complete=True,
                      metrics={arm: {'foreground_miou': 0.62} for arm in ('grid_probq', *source.ARMS)},
                      full_validation_evidence=step == 5000,
                      final_integrity_pending=step != 5000)
        folder = 'final_val' if step == 5000 else 'val_step_%06d' % step
        self.write(self.train / folder / 'result.json', report)

    def check(self, step=500):
        return source.validate_source(self.root, self.tool, step)

    def mutate_json(self, path, **changes):
        data = source.read_json(path)
        data.update(changes)
        self.write(path, data)

    def test_three_original_checkpoints_are_read_only(self):
        before = {str(p): (source.digest(p), p.stat().st_mtime_ns)
                  for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        for step in (500, 1500, 5000):
            static = self.check(step)
            self.assertEqual(static['meta']['step'], step)
            self.assertTrue(source.check_source_unchanged(static))
        after = {str(p): (source.digest(p), p.stat().st_mtime_ns)
                 for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_validation_does_not_import_torch(self):
        original = __import__
        def deny_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('Static validation imported Torch')
            return original(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=deny_torch):
            self.check()

    def test_bad_checkpoint_hash(self):
        (self.train / 'checkpoints/checkpoint_000500.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'binding failed'):
            self.check()

    def test_partial_or_smoke_run_rejected(self):
        path = self.root / 'workflow_state.json'
        self.mutate_json(path, complete=False)
        with self.assertRaisesRegex(ValueError, 'formal run'):
            self.check()
        self.mutate_json(path, complete=True, smoke=True)
        with self.assertRaisesRegex(ValueError, 'formal run'):
            self.check()

    def test_train_val_overlap_rejected(self):
        self.manifest['val_names'][0] = self.manifest['train_names'][0]
        self.write(self.train / 'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.check()

    def test_checkpoint_protocol_and_completion_binding(self):
        path = self.train / 'checkpoints/checkpoint_000500.pt.json'
        self.mutate_json(path, protocol_info={'execution_settings': {}})
        with self.assertRaisesRegex(ValueError, 'protocol bindings'):
            self.check()
        self.mutate_json(path, protocol_info=self.protocol, training_complete=True)
        with self.assertRaisesRegex(ValueError, 'binding failed'):
            self.check()

    def test_final_report_requires_final_integrity(self):
        path = self.train / 'final_val/result.json'
        self.mutate_json(path, final_integrity_pending=True)
        with self.assertRaisesRegex(ValueError, 'final Val450 integrity'):
            self.check(5000)

    def test_missing_arm_reference_rejected(self):
        path = self.train / 'val_step_000500/result.json'
        self.mutate_json(path, metrics={'grid_probq': {}, 'd': {}})
        with self.assertRaisesRegex(ValueError, 'Val450'):
            self.check()

    def test_release_rejects_unbound_python(self):
        (self.tool / 'new_helper.py').write_text('# unbound\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected Python'):
            self.check()

    def test_release_rejects_changed_file_and_symlink(self):
        path = self.tool / 'README.md'
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            self.check()
        path.unlink()
        outside = Path(self.temporary.name) / 'readme'
        outside.write_text('synthetic fixture\n')
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            self.check()

    def test_checksum_manifest_does_not_override_expected_source(self):
        path = self.tool / 'common.py'
        path.write_text('# changed source with a self-consistent release record\n')
        checksums = source.read_json(self.tool / 'SHA256SUMS.json')
        checksums['common.py'] = source.digest(path)
        self.write(self.tool / 'SHA256SUMS.json', checksums)
        with self.assertRaisesRegex(ValueError, 'exact original fix2'):
            self.check()

    def test_postaudit_source_changes_detected(self):
        static = self.check()
        (self.train / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            source.check_source_unchanged(static)

    def test_import_collision_fails_before_torch_import(self):
        static = self.check()
        unrelated = SimpleNamespace(__name__='common', __file__=__file__)
        with mock.patch.dict(sys.modules, {'common': unrelated}):
            with self.assertRaisesRegex(ValueError, 'wrong historical module'):
                source.import_runtime(static)

    def test_runtime_arguments_copy_source_and_reject_source_output(self):
        saved = {'config': self.config}
        original = json.dumps(saved, sort_keys=True)
        output = Path(self.temporary.name) / 'audit'
        args = source.runtime_arguments(saved, output)
        self.assertEqual(args.output, str(output))
        self.assertFalse(args.allow_smoke_values)
        self.assertEqual(args.protocol, self.config['protocol'])
        self.assertEqual(json.dumps(saved, sort_keys=True), original)
        for bad in (self.train / 'audit', self.root / 'sibling', Path(self.config['values']) / 'audit'):
            with self.assertRaises(ValueError):
                source.runtime_arguments(saved, bad)

    def test_runtime_ledger_rejects_unbound_and_changed_inputs(self):
        saved = {'protocol_info': self.protocol, 'input_file_sha256': {'/a': 'original', '/b': 'original'}}
        runtime = SimpleNamespace(protocol_info=self.protocol, input_ledger={Path('/a'): 'original'})
        self.assertTrue(source.check_runtime_inputs(runtime, saved))
        runtime.input_ledger[Path('/a')] = 'changed'
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            source.check_runtime_inputs(runtime, saved)
        runtime.input_ledger = {Path('/unseen'): 'original'}
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            source.check_runtime_inputs(runtime, saved)

    def test_bad_steps(self):
        for step in (True, 0, 1000, 500.0, '500'):
            with self.assertRaisesRegex(ValueError, 'Audit step'):
                self.check(step)

    def torch_fixture(self, step=500):
        import torch
        # Independent test hash: verifies delegation to the supplied original
        # common module instead of reimplementing its serialization algorithm.
        def test_hash(module):
            data = [(k, str(v.dtype), list(v.shape), v.tolist())
                    for k, v in module.state_dict().items()]
            return hashlib.sha256(json.dumps(data).encode()).hexdigest()
        head = torch.nn.Linear(2, 4)
        payload = dict(format=source.SOURCE_FORMAT, step=step, mode='train', smoke=False,
                       training_complete=step == 5000, source_hash=self.expected,
                       protocol_info=self.protocol, config=self.config,
                       heads={arm: head.state_dict() for arm in source.ARMS},
                       input_file_sha256={'/fixture/input': 'fixture-hash'},
                       updates_per_arm={arm: step for arm in source.ARMS})
        filename = 'checkpoint_last.pt' if step == 5000 else 'checkpoint_%06d.pt' % step
        checkpoint = self.train / 'checkpoints' / filename
        torch.save(payload, checkpoint)
        self.mutate_json(checkpoint.with_suffix('.pt.json'), sha256=source.digest(checkpoint),
                         head_hashes={a: test_hash(head) for a in source.ARMS})
        modules = SimpleNamespace(torch=torch, common=SimpleNamespace(module_hash=test_hash))
        return payload, modules, checkpoint

    @unittest.skipUnless(HAS_TORCH, 'CPU Torch is required for checkpoint fixture tests')
    def test_cpu_payload_does_not_need_optimizer_or_cuda_rng(self):
        for step in (500, 1500, 5000):
            _, modules, _ = self.torch_fixture(step)
            saved = source.load_saved(self.check(step), modules)
            self.assertEqual(saved['step'], step)
            self.assertNotIn('rng', saved)
            self.assertNotIn('optimizers', saved)

    @unittest.skipUnless(HAS_TORCH, 'CPU Torch is required for checkpoint fixture tests')
    def test_payload_provenance_mismatch(self):
        payload, modules, checkpoint = self.torch_fixture()
        payload['updates_per_arm']['b'] = 499
        modules.torch.save(payload, checkpoint)
        self.mutate_json(checkpoint.with_suffix('.pt.json'), sha256=source.digest(checkpoint))
        with self.assertRaisesRegex(ValueError, 'training provenance'):
            source.load_saved(self.check(), modules)

    @unittest.skipUnless(HAS_TORCH, 'CPU Torch is required for checkpoint fixture tests')
    def test_payload_config_must_match_manifest_arguments(self):
        payload, modules, checkpoint = self.torch_fixture()
        payload['config'] = dict(payload['config'], chunk_size=999)
        modules.torch.save(payload, checkpoint)
        self.mutate_json(checkpoint.with_suffix('.pt.json'), sha256=source.digest(checkpoint))
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            source.load_saved(self.check(), modules)

    @unittest.skipUnless(HAS_TORCH, 'CPU Torch is required for checkpoint fixture tests')
    def test_payload_rejects_nonfinite_or_changed_head_tensors(self):
        payload, modules, checkpoint = self.torch_fixture()
        payload['heads']['a']['weight'][0, 0] += 10
        modules.torch.save(payload, checkpoint)
        self.mutate_json(checkpoint.with_suffix('.pt.json'), sha256=source.digest(checkpoint))
        with self.assertRaisesRegex(ValueError, 'head tensor digest'):
            source.load_saved(self.check(), modules)
        payload['heads']['a']['weight'][0, 0] = float('nan')
        modules.torch.save(payload, checkpoint)
        self.mutate_json(checkpoint.with_suffix('.pt.json'), sha256=source.digest(checkpoint))
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            source.load_saved(self.check(), modules)


if __name__ == '__main__':
    unittest.main()
