"""Read-only comparison contracts, with real published metric arithmetic."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import compare_recovery as subject
from recovery import CHECKPOINT_SHA


class ProtocolContracts(unittest.TestCase):
    def test_only_explicit_blocking_change_is_permitted(self):
        before = dict(bank='same', execution_settings=dict(cuda_launch_blocking=None, dtype='fp32'))
        after = copy.deepcopy(before)
        after['execution_settings']['cuda_launch_blocking'] = '1'
        snapshot = copy.deepcopy(before)
        result = subject.checked_protocols(before, after)
        self.assertEqual(result['execution_settings.cuda_launch_blocking'], {'before': None, 'after': '1'})
        self.assertEqual(before, snapshot)
        for key in ('bank', 'execution_settings'):
            bad = copy.deepcopy(after)
            if key == 'bank':
                bad[key] = 'different'
            else:
                bad[key]['dtype'] = 'fp16'
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.checked_protocols(before, bad)
        with self.assertRaises(ValueError):
            subject.checked_protocols(before, before)


class ComparisonContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = Path(os.environ.get('BRIGHT_READOUT_TEST_PACKAGE',
            '/workspace/bright_next_step_publish_remote_20261021/research/bright_memory_readout_design_20261022'))
        if not cls.package.is_dir():
            raise unittest.SkipTest('Set BRIGHT_READOUT_TEST_PACKAGE to the pinned installed readout package')
        cls.helper = subject.load_analysis(cls.package)

    def fixture(self, root, variant, blocking, recovery=False):
        root.mkdir()
        folder = root / 'val_step_002500'
        folder.mkdir()
        names = ['image_%04d' % i for i in range(450)]
        grid = [[100, 2, 1, 0], [3, 20, 2, 1], [1, 4, 10, 1], [1, 1, 2, 12]]
        cm = copy.deepcopy(grid)
        if recovery:
            cm[2][1] -= 1; cm[2][2] += 1
        matrices = {'grid_probq': grid, variant: cm}
        rows = [dict(image=n, event='event_'+str(i % 2), confusion=matrices) for i,n in enumerate(names)]
        (folder / 'images.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        total = {key: [[450*x for x in row] for row in value] for key,value in matrices.items()}
        report = dict(format='BRIGHT_MEMORY_READOUT_DESIGN_V1_EVALUATION', complete=True, phase='complete',
                      final_integrity_pending=False, smoke=False, split='val', step=2500, seed=3407,
                      image_count=450, requested_names=names, variant=variant,
                      class_names=['background','intact','damaged','destroyed'],
                      metrics={key:dict(self.helper.metrics(value),confusion=value) for key,value in total.items()})
        plan = {key:'same' for key in ('source','addon_sha256','extension_files','post500_lr_multiplier',
                'scheduler_horizon','warmup_steps','maximum_lr','weight_decay','grad_clip','optimizer','endpoint_step')}
        plan.update(variant=variant, read_weight=.01 if recovery else 0)
        manifest = dict(initial_head_sha256='same',train_names=['train'],val_names=names,
                        actual_training_names=['train'],fixed_train_names=['train'],parameter_count=10,support_arm='d',
                        protocol_info=dict(bank='same',execution_settings=dict(cuda_launch_blocking=blocking)),
                        runtime_config=dict(output=str(root),setting='same'),plan=plan)
        if recovery:
            lineage = dict(restored_step=800,checkpoint_sha256=CHECKPOINT_SHA,inherited_training_rows=800,
                           new_training_starts_at=801,exact_historical_replay=False,historical_failure_resolved=False)
            manifest['recovery'] = lineage
            report['recovery'] = lineage
        summary = dict(phase='complete',complete=True,smoke=False,training_complete=True,
                       full_validation_evidence=True,last_complete_step=2500,restored_step=800,
                       new_updates=1700,test_accessed=False)
        for path, value in ((root/'manifest.json',manifest),(root/'summary.json',summary),(folder/'result.json',report)):
            path.write_text(json.dumps(value))
        return folder/'result.json'

    def test_numeric_parity_and_provenance_and_no_report_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = self.fixture(root/'before','original','1')
            after = self.fixture(root/'after','read_aux','1',True)
            expected, expected_rows = self.helper.compare(before,after)
            original_bytes = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result, output = subject.compare(self.package,root/'before',root/'after')
            self.assertEqual(result['differences'],expected['differences'])
            self.assertEqual(result['per_event'],expected['per_event'])
            self.assertEqual(result['grid_confusion_identical'],expected['grid_confusion_identical'])
            self.assertFalse(result['exact_historical_trajectory_match'])
            self.assertTrue(result['exact_execution_protocol_match'])
            for path, data in original_bytes.items():
                self.assertEqual(path.read_bytes(),data)
            self.assertTrue((output/'paired_images.csv').is_file())
            with self.assertRaises(FileExistsError):
                subject.compare(self.package,root/'before',root/'after')

    def test_blocking_difference_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root/'before','original',None)
            self.fixture(root/'after','read_aux','1',True)
            result,_ = subject.compare(self.package,root/'before',root/'after')
            self.assertFalse(result['exact_execution_protocol_match'])
            self.assertIsNone(result['before_protocol']['execution_settings']['cuda_launch_blocking'])
            self.assertEqual(result['after_protocol']['execution_settings']['cuda_launch_blocking'],'1')

    def test_rejects_incomplete_recovery_and_other_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root/'before','original',None)
            self.fixture(root/'after','read_aux','1',True)
            summarypath = root/'after/summary.json'
            summary = json.loads(summarypath.read_text())
            summary['training_complete'] = False
            summarypath.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                subject.compare(self.package,root/'before',root/'after')
            summary['training_complete'] = True
            summarypath.write_text(json.dumps(summary))
            manifestpath = root/'after/manifest.json'
            manifest = json.loads(manifestpath.read_text())
            manifest['protocol_info']['bank'] = 'other'
            manifestpath.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                subject.compare(self.package,root/'before',root/'after')
            self.assertFalse((root/'after/comparison_original_to_read_aux').exists())


if __name__ == '__main__':
    unittest.main()
