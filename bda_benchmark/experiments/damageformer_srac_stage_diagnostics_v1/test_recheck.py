"""Failed-image replay entry contracts; no BRIGHT data or CUDA needed."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import recheck_feature as module


class RecheckContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failed = self.root/'failed'
        self.failed.mkdir()
        self.output = self.root/'new'
        self.val = ['image_'+str(i) for i in range(450)]
        self.original = dict(checkpoint=str(self.root/'snapshot.pth'), a1_checkpoint=str(self.root/'a1.pth'),
                             dataset='unchanged_dataset', train_list=str(self.root/'train_set.txt'),
                             val_list=str(self.root/'val_set.txt'), output=str(self.failed),
                             device='cuda:0', workers=4, seed=3407, smoke=False,
                             start_index=0, max_images=0, embedding_interventions=True)
        self.summary = dict(complete=False, audit_type='feature', failure={'image': self.val[72]})
        self.manifest = dict(arguments=copy.deepcopy(self.original), audit_type='feature',
                             selected_images=self.val, checkpoint_sha256='hash', a1_sha256='hash',
                             train_sha256='hash', val_sha256='hash', snapshot_metadata_sha256='hash',
                             stage_source_sha256='old_code_hash')
        self.lists = patch.object(module, 'lists', return_value=(['train'], self.val)).start()
        self.digest = patch.object(module, 'digest', return_value='hash').start()
        self.addCleanup(patch.stopall)

    def write(self):
        (self.failed/'summary.json').write_text(json.dumps(self.summary), encoding='utf-8')
        (self.failed/'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def load(self, prefix=False):
        self.write()
        return module.load_recheck_arguments(argparse.Namespace(
            failed_run=str(self.failed), output=str(self.output), prefix=prefix))

    def test_single_preserves_original_settings_and_old_files(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.failed.iterdir()}
        actual = self.load()
        self.assertEqual(actual.start_index, 72)
        self.assertEqual(actual.max_images, 1)
        self.assertFalse(actual.smoke)
        for field in ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list',
                      'device', 'workers', 'seed', 'embedding_interventions'):
            self.assertEqual(getattr(actual, field), self.original[field])
        self.assertEqual(actual.recheck_origin['original_stage_source_sha256'], 'old_code_hash')
        self.assertEqual(actual.recheck_origin['mode'], 'single_image')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.failed.iterdir()})
        self.assertFalse(self.output.exists())

    def test_prefix_uses_full_val_order_not_progress_log(self):
        self.manifest['arguments'].update(start_index=60, max_images=40)
        self.manifest['selected_images'] = self.val[60:100]
        actual = self.load(prefix=True)
        self.assertEqual((actual.start_index, actual.max_images), (0, 73))
        self.assertEqual(actual.recheck_origin['original_selected_index_zero_based'], 12)

    def test_completed_or_wrong_audit_rejected(self):
        for field, value in (('complete', True), ('audit_type', 'memory')):
            original = self.summary[field]
            self.summary[field] = value
            with self.assertRaises(ValueError):
                self.load()
            self.summary[field] = original

    def test_unknown_or_missing_failure_image_rejected(self):
        for failure in ({}, {'image': 'unknown'}, None):
            self.summary['failure'] = failure
            with self.assertRaises(ValueError):
                self.load()

    def test_cannot_write_into_old_run_or_existing_output(self):
        for output in (self.failed, self.failed/'nested', self.root):
            self.output = output
            with self.assertRaises((ValueError, FileExistsError)):
                self.load()

    def test_missing_original_arguments_hash_or_selected_images_rejected(self):
        baseline = copy.deepcopy(self.manifest)
        for key in ('arguments', 'selected_images', 'checkpoint_sha256', 'a1_sha256',
                    'train_sha256', 'val_sha256'):
            self.manifest = copy.deepcopy(baseline)
            del self.manifest[key]
            with self.assertRaises(ValueError):
                self.load()
        self.manifest = copy.deepcopy(baseline)
        del self.manifest['arguments']['workers']
        with self.assertRaises(ValueError):
            self.load()

    def test_input_hash_mismatch_rejected(self):
        self.manifest['train_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'hash mismatch: train_list'):
            self.load()

    def test_stale_selected_images_or_wrong_output_rejected(self):
        self.manifest['selected_images'] = self.val[::-1]
        with self.assertRaisesRegex(ValueError, 'selected_images'):
            self.load()
        self.manifest['selected_images'] = self.val
        self.manifest['arguments']['output'] = str(self.root/'another_run')
        with self.assertRaisesRegex(ValueError, 'Manifest output'):
            self.load()

    def test_missing_run_or_manifest_rejected(self):
        args = argparse.Namespace(failed_run=str(self.root/'absent'), output=str(self.output), prefix=False)
        with self.assertRaisesRegex(ValueError, 'directory is missing'):
            module.load_recheck_arguments(args)
        args.failed_run = str(self.failed)
        with self.assertRaisesRegex(ValueError, 'both summary.json and manifest.json'):
            module.load_recheck_arguments(args)

    def test_cli_has_no_training_or_checkpoint_override(self):
        args = module.parser().parse_args(['--failed_run', 'old', '--output', 'new', '--prefix'])
        self.assertEqual(set(vars(args)), {'failed_run', 'output', 'prefix'})
        self.assertTrue(args.prefix)


if __name__ == '__main__':
    unittest.main()
