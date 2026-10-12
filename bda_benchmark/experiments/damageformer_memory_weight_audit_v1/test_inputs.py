"""Parent-audit input recovery must not silently switch experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from .inputs import load_parent, verify_parent_state, assert_parent_unchanged
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parent = self.root/'previous'
        self.parent.mkdir()
        args = dict(source_head='hard', pool_mode='hard', read_mode='hard')
        for name in ('checkpoint', 'train_list', 'val_list'):
            path = self.root/name
            path.write_text('trusted synthetic fixture', encoding='utf-8')
            args[name] = str(path)
        for name in ('run_root', 'memory_run', 'bank', 'dataset'):
            path = self.root/name
            path.mkdir()
            args[name] = str(path)
        self.summary = dict(format='MEMORY_READOUT_AUDIT_V1', complete=True, smoke=False,
            test_accessed=False, image_count=450, source_verified=True, bank_verified=True,
            head_verified=True, reproduction=dict(passed=True), final_metrics={})
        self.manifest = dict(format='MEMORY_READOUT_AUDIT_V1', smoke=False, test_accessed=False,
            arguments=args, input_file_sha256={args['checkpoint']: common.digest(args['checkpoint'])},
            source_identity=dict(dim=128), settings=dict(seed=3407, topk=8))
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        for name, data in (('summary.json', self.summary), ('manifest.json', self.manifest)):
            (self.parent/name).write_text(json.dumps(data), encoding='utf-8')

    def load(self, **kwargs):
        return load_parent(str(self.parent), str(self.root/'new_audit'), **kwargs)

    def test_recovers_paths_and_runtime_overrides_only(self):
        args, parent = self.load(smoke=True, workers=0, device='cpu')
        self.assertEqual(args.checkpoint, self.manifest['arguments']['checkpoint'])
        self.assertEqual(args.memory_run, self.manifest['arguments']['memory_run'])
        self.assertEqual(args.source_head, 'hard')
        self.assertTrue(args.smoke)
        self.assertEqual(args.device, 'cpu')
        state = dict(files=parent['expected_inputs'], identity=parent['source_identity'], settings=parent['settings'])
        verify_parent_state(parent, state)

    def test_smoke_parent_never_allowed_even_for_smoke(self):
        self.summary['smoke'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'formal'):
            self.load(smoke=True)

    def test_incomplete_unverified_or_bad_reproduction_fails(self):
        for key in ('complete', 'source_verified', 'bank_verified', 'head_verified'):
            with self.subTest(key=key):
                self.summary[key] = False
                self.save()
                with self.assertRaises(ValueError): self.load()
                self.summary[key] = True
        self.summary['reproduction'] = dict(passed=False)
        self.save()
        with self.assertRaises(ValueError): self.load()

    def test_input_hash_mutation_fails(self):
        Path(self.manifest['arguments']['checkpoint']).write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash changed'): self.load()

    def test_missing_original_input_does_not_fall_back(self):
        self.manifest['arguments']['checkpoint'] = str(self.root/'absent')
        self.save()
        with self.assertRaisesRegex(FileNotFoundError, 'do not substitute'): self.load()

    def test_output_cannot_be_in_old_source_or_parent(self):
        for path in (self.parent/'new', Path(self.manifest['arguments']['bank'])/'new'):
            with self.assertRaisesRegex(ValueError, 'isolated'):
                load_parent(str(self.parent), str(path))

    def test_output_exists_not_overwritten(self):
        (self.root/'new_audit').mkdir()
        with self.assertRaises(FileExistsError): self.load()

    def test_parent_mutation_detected_at_end(self):
        _, parent = self.load()
        self.summary['new'] = 1
        self.save()
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            assert_parent_unchanged(parent)

    def test_loaded_wrong_files_or_settings_fail(self):
        _, parent = self.load()
        state = dict(files={}, identity=parent['source_identity'], settings=parent['settings'])
        with self.assertRaisesRegex(ValueError, 'exact inputs'): verify_parent_state(parent, state)
        state['files'] = parent['expected_inputs']
        state['settings'] = dict(topk=16)
        with self.assertRaisesRegex(ValueError, 'settings'): verify_parent_state(parent, state)

    def test_nonabsolute_output_or_negative_workers_fails(self):
        with self.assertRaises(ValueError): load_parent(str(self.parent), 'relative')
        with self.assertRaises(ValueError): self.load(workers=-1)


if __name__ == '__main__': unittest.main()
