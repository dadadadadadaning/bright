"""Stdlib contracts: calibration804 provenance and unpromoted Val450 evidence."""
import builtins
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import helpers
import preflight


EXPERIMENT = '''import hashlib
import importlib.util
from pathlib import Path
VERIFIED = False
def load_file(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def verify_plan(plan, bridge):
    global VERIFIED
    if bridge.check_package(plan['addon_dir']) != plan['addon_sha256']:
        raise ValueError('Addon source package changed')
    bridge.verify_ledgers(plan['source'])
    VERIFIED = True
'''
BRIDGE = '''import hashlib
from pathlib import Path
def check_package(package):
    return hashlib.sha256((Path(package)/'SHA256SUMS.json').read_bytes()).hexdigest()
def verify_ledgers(source):
    for path, expected in source['ledgers'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Source ledger changed')
'''


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def diagonal(count):
    return [[count if i == j else 0 for j in range(4)] for i in range(4)]


def paired(count):
    return [[[count if gt == old == new else 0 for new in range(4)]
             for old in range(4)] for gt in range(4)]


class PreflightContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root/'package'
        self.parent = self.root/'night_3407_z_yxvq4s/train_calibration'
        self.addon = self.root/'addon'
        for directory in (self.package, self.parent/'checkpoints', self.addon):
            directory.mkdir(parents=True)
        (self.package/'experiment.py').write_text(EXPERIMENT)
        (self.package/'old_model_adapter.py').write_text('# fixture old adapter\n')
        (self.package/'severity_adapter.py').write_text('# fixture severity adapter\n')
        self.identity = {name: helpers.sha(self.package/name) for name in preflight.IDENTITY}
        (self.addon/'runtime_bridge.py').write_text(BRIDGE)
        dump(self.addon/'SHA256SUMS.json', {'runtime_bridge.py': helpers.sha(self.addon/'runtime_bridge.py')})
        self.ledger = self.root/'source_manifest.json'
        dump(self.ledger, {'original_source': 'unchanged'})
        self.checkpoint = self.parent/'checkpoints/checkpoint_000800.pt'
        self.checkpoint.write_bytes(b'Not a pickle. Preflight must only hash these bytes.')
        self.checkpoint_sha = helpers.sha(self.checkpoint)
        self.plan = dict(format='BRIGHT_SEVERITY_MEMORY_V1', variant='calibration',
            effective_mode='calibration', seed=3407, endpoint_step=2500, read_weight=0,
            extension_files=self.identity, scheduler_horizon=5000, maximum_lr=.0001,
            weight_decay=.005, warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
            post500_lr_multiplier=.2, fixed_train_count=450, test_accessed=False,
            fresh_initialization=True, old_head_weights_reused=False, historical_trajectory_claim=False,
            added_loss=None, original_loss='exact runtime.loss (NLL plus Lovasz), unchanged',
            addon_dir=str(self.addon), addon_sha256=helpers.sha(self.addon/'SHA256SUMS.json'),
            source={'ledgers': {str(self.ledger): helpers.sha(self.ledger)}})
        self.train_names = [f'train_{index:05d}' for index in range(2918)]
        self.train_names[803] = preflight.TARGET_IMAGE
        self.val_names = [f'val_{index:05d}' for index in range(450)]
        self.manifest = dict(format='BRIGHT_SEVERITY_MEMORY_V1', plan=self.plan, seed=3407,
            train_names=self.train_names, val_names=self.val_names, actual_training_names=self.train_names,
            fixed_train_names=sorted(random.Random(3407+191).sample(self.train_names, 450)),
            support_arm='d', fresh_initialization=True, test_accessed=False,
            runtime_config={'device': 'cuda:0', 'workers': 1}, protocol_info={'fixture': True})
        checkpoint_record = dict(path=str(self.checkpoint), step=800, sha256=self.checkpoint_sha,
                                 roundtrip_verified=True, variant='calibration')
        dump(self.checkpoint.with_suffix('.pt.json'), checkpoint_record)
        self.status = dict(format='BRIGHT_SEVERITY_MEMORY_V1', variant='calibration', effective_mode='calibration',
            seed=3407, phase='failed', complete=False, training_complete=False, full_validation_evidence=False,
            smoke=False, test_accessed=False, last_complete_step=803, endpoint_step=2500,
            failure_phase='update_804', current_image=preflight.TARGET_IMAGE, last_checkpoint=checkpoint_record,
            error='RuntimeError: CUDA error: device-side assert triggered')
        dump(self.parent/'summary.json', self.status)
        dump(self.parent/'first_failure.json', dict(phase='update_804', image=preflight.TARGET_IMAGE,
            last_complete_step=803, tensors_modified=False, automatic_retry=False,
            error=self.status['error'], traceback='loss.backward()'))
        self.rows = [dict(step=step, optimizer_step=step, epoch=0, epoch_seed=3407,
                         variant='calibration', image=self.train_names[step-1], event='training_event',
                         original_loss=.5, training_loss=.5, grad_norm_preclip=.2, lr=.0001,
                         read_loss=0., read_loss_evaluated=False) for step in range(1, 804)]
        self.write_rows()
        self.write_plan()
        for folder, step in (('initial_val', 0), ('val_step_000500', 500)):
            self.write_evaluation(folder, step)

    def write_plan(self):
        dump(self.parent/'plan.json', self.plan)
        self.manifest['plan'] = self.plan
        dump(self.parent/'manifest.json', self.manifest)

    def write_rows(self):
        (self.parent/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))

    def write_evaluation(self, folder, step):
        directory = self.parent/folder
        directory.mkdir()
        rows = [dict(index=index, image=name, event='event_'+str(index % 2), test_accessed=False,
                     confusion={key: diagonal(1) for key in ('grid_probq', 'calibration')},
                     original_loss={'grid_probq': 1., 'calibration': .5},
                     paired_counts_grid_to_variant=paired(1)) for index, name in enumerate(self.val_names)]
        journal = directory/'images.jsonl'
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report = dict(format='BRIGHT_SEVERITY_MEMORY_V1_EVALUATION', step=step, split='val', seed=3407,
            variant='calibration', image_count=450, requested_names=self.val_names, execution_complete=True,
            journal_verified=True, whole_val450_processed=True, smoke=False, test_accessed=False,
            head_state_unchanged=True, rng_restored=True, phase='evaluation_complete_integrity_pending',
            complete=False, full_validation_evidence=False, final_integrity_pending=True,
            images_sha256=helpers.sha(journal),
            metrics={key: dict(confusion=diagonal(450), mean_original_loss=loss)
                     for key, loss in (('grid_probq', 1.), ('calibration', .5))},
            per_event={'event_'+str(i): {key: {'confusion': diagonal(225)} for key in ('grid_probq', 'calibration')}
                       for i in range(2)}, paired_counts_grid_to_variant=paired(450),
            paired_axes=['GT', 'grid_probq', 'calibration'])
        dump(directory/'result.json', report)

    def mutate(self, path, **updates):
        value = helpers.read(path)
        value.update(updates)
        dump(path, value)

    def check(self):
        with patch.object(preflight, 'PARENT', self.parent), \
             patch.object(preflight, 'IDENTITY', self.identity), \
             patch.object(preflight, 'CHECKPOINT_SHA256', self.checkpoint_sha):
            return preflight.check_inputs(self.package, self.parent)

    def file_hashes(self):
        return {str(path): helpers.sha(path) for path in self.root.rglob('*') if path.is_file()}

    def test_returns_exactly_seven_values_and_verifies_original_source_plan(self):
        exp, plan, manifest, rows, checkpoint, fingerprints, info = self.check()
        self.assertTrue(exp.VERIFIED)
        self.assertEqual(plan, self.plan)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(rows, self.rows)
        self.assertEqual(checkpoint, self.checkpoint)
        self.assertEqual(fingerprints[str(checkpoint)], self.checkpoint_sha)
        self.assertEqual(fingerprints[str(self.ledger)], helpers.sha(self.ledger))
        self.assertEqual(set(info['inherited_evaluations']), {'initial_val', 'val_step_000500'})
        self.assertEqual(len(info['fingerprints']), 4)
        self.assertFalse(info['prior_replay_required'])

    def test_preserves_both_pending_evaluations_and_all_parent_bytes(self):
        # Suppress normal import bytecode writes when checking the read-only gate.
        with patch('sys.dont_write_bytecode', True):
            before = self.file_hashes()
            info = self.check()[-1]
            after = self.file_hashes()
        self.assertEqual(before, after)
        for folder, reference in info['inherited_evaluations'].items():
            original = helpers.read(self.parent/folder/'result.json')
            for key in ('phase', 'complete', 'full_validation_evidence', 'final_integrity_pending'):
                self.assertEqual(reference[key], original[key])
            self.assertFalse(reference['complete'])
            self.assertTrue(reference['final_integrity_pending'])
            self.assertTrue(reference['inherited_reference_only'])
            self.assertFalse(reference['promoted_by_recovery'])

    def test_rejects_a_different_parent_even_if_evidence_is_valid(self):
        with patch.object(preflight, 'PARENT', self.root/'different_parent'):
            with self.assertRaisesRegex(ValueError, 'fixed failed calibration parent'):
                preflight.check_inputs(self.package, self.parent)

    def test_rejects_changed_installed_original_source(self):
        (self.package/'severity_adapter.py').write_text('# altered source\n')
        with self.assertRaisesRegex(ValueError, 'Unsupported installed version'):
            self.check()

    def test_rejects_changed_recipe_with_matching_manifest_plan(self):
        self.plan['variant'] = 'severity_memory'
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'parent recipe: variant'):
            self.check()

    def test_rejects_changed_checkpoint_even_with_rehashed_sidecar(self):
        self.checkpoint.write_bytes(b'Changed checkpoint')
        changed = helpers.sha(self.checkpoint)
        self.mutate(self.checkpoint.with_suffix('.pt.json'), sha256=changed)
        self.status['last_checkpoint']['sha256'] = changed
        dump(self.parent/'summary.json', self.status)
        with self.assertRaisesRegex(ValueError, 'Pinned checkpoint800 SHA'):
            self.check()

    def test_rejects_unverified_checkpoint_roundtrip(self):
        self.mutate(self.checkpoint.with_suffix('.pt.json'), roundtrip_verified=False)
        with self.assertRaisesRegex(ValueError, 'Checkpoint800 sidecar'):
            self.check()

    def test_rejects_summary_checkpoint_not_matching_sidecar(self):
        self.status['last_checkpoint']['step'] = 750
        dump(self.parent/'summary.json', self.status)
        with self.assertRaisesRegex(ValueError, 'Checkpoint800 summary'):
            self.check()

    def test_rejects_wrong_failure_image(self):
        self.mutate(self.parent/'first_failure.json', image='another_image')
        with self.assertRaisesRegex(ValueError, 'First failure'):
            self.check()

    def test_rejects_wrong_last_complete_step(self):
        self.mutate(self.parent/'summary.json', last_complete_step=802)
        with self.assertRaisesRegex(ValueError, 'calibration804 failure: last_complete_step'):
            self.check()

    def test_rejects_partial_parent_journal(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'exactly803'):
            self.check()

    def test_rejects_nonconsecutive_optimizer_step(self):
        self.rows[800]['optimizer_step'] = 800
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
            self.check()

    def test_rejects_auxiliary_loss_even_if_it_is_finite(self):
        self.rows[800]['training_loss'] += .1
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'objective/scalars'):
            self.check()

    def test_rejects_smoke_or_partial_train_scope(self):
        self.manifest['actual_training_names'] = self.train_names[:3]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'manifest scope'):
            self.check()

    def test_rejects_train_val_overlap(self):
        self.val_names[0] = self.train_names[0]
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'Train/Val overlap'):
            self.check()

    def test_rejects_changed_fixed_train_subset(self):
        self.manifest['fixed_train_names'] = list(reversed(self.manifest['fixed_train_names']))
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'Fixed Train subset'):
            self.check()

    def test_rejects_changed_source_ledger_before_import(self):
        dump(self.ledger, {'original_source': 'changed'})
        with self.assertRaisesRegex(ValueError, 'Source ledger changed'):
            self.check()

    def test_calls_original_verify_plan_to_reject_changed_addon(self):
        dump(self.addon/'SHA256SUMS.json', {'changed': 'release'})
        with self.assertRaisesRegex(ValueError, 'Addon source package changed'):
            self.check()

    def test_rejects_partial_validation_flag(self):
        self.mutate(self.parent/'val_step_000500/result.json', whole_val450_processed=False)
        with self.assertRaisesRegex(ValueError, 'whole_val450_processed'):
            self.check()

    def test_rejects_unverified_validation_journal(self):
        self.mutate(self.parent/'initial_val/result.json', journal_verified=False)
        with self.assertRaisesRegex(ValueError, 'journal_verified'):
            self.check()

    def test_rejects_promoted_parent_evaluation(self):
        self.mutate(self.parent/'val_step_000500/result.json', complete=True)
        with self.assertRaisesRegex(ValueError, 'pending flag differs: complete'):
            self.check()

    def test_rejects_evaluation_journal_hash_change(self):
        journal = self.parent/'val_step_000500/images.jsonl'
        journal.write_text(journal.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'journal hash'):
            self.check()

    def change_evaluation_rows(self, change):
        journal = self.parent/'val_step_000500/images.jsonl'
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        change(rows)
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.mutate(journal.with_name('result.json'), images_sha256=helpers.sha(journal))

    def test_rejects_missing_evaluation_image_even_with_recomputed_hash(self):
        self.change_evaluation_rows(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Val coverage'):
            self.check()

    def test_rejects_changed_evaluation_order_even_with_recomputed_hash(self):
        self.change_evaluation_rows(lambda rows: rows[1].update(index=0))
        with self.assertRaisesRegex(ValueError, 'journal order'):
            self.check()

    def test_rejects_confusion_inconsistent_with_paired_counts(self):
        self.change_evaluation_rows(lambda rows: rows[1]['confusion']['calibration'][0].__setitem__(0, 2))
        with self.assertRaisesRegex(ValueError, 'paired/confusion'):
            self.check()

    def test_rejects_changed_aggregate_confusion(self):
        path = self.parent/'val_step_000500/result.json'
        report = helpers.read(path)
        report['metrics']['calibration']['confusion'][0][0] += 1
        dump(path, report)
        with self.assertRaisesRegex(ValueError, 'aggregate confusion'):
            self.check()

    def test_rejects_changed_event_confusion(self):
        path = self.parent/'val_step_000500/result.json'
        report = helpers.read(path)
        report['per_event']['event_0']['calibration']['confusion'][0][0] += 1
        dump(path, report)
        with self.assertRaisesRegex(ValueError, 'event confusion'):
            self.check()

    def test_rejects_nonfinite_journal_loss(self):
        self.change_evaluation_rows(lambda rows: rows[0]['original_loss'].__setitem__('calibration', float('nan')))
        with self.assertRaisesRegex(ValueError, 'nonfinite loss'):
            self.check()

    def test_stdlib_preflight_never_imports_torch_or_deserializes_checkpoint(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('Torch imported in stdlib preflight')
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=guarded_import), \
             patch('pickle.load', side_effect=AssertionError('Checkpoint deserialized')), \
             patch('pickle.loads', side_effect=AssertionError('Checkpoint deserialized')):
            self.check()
        self.assertTrue(self.checkpoint.read_bytes().startswith(b'Not a pickle'))


if __name__ == '__main__':
    unittest.main()
