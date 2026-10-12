"""CPU tests of provenance rejection and predeclared metric gates, using tiny heads."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from . import provenance as p
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


class TrainedFixture:
    def __init__(self, root, smoke=False, bank_names=None):
        self.run, self.bank = root / 'trained', root / 'bank'
        self.run.mkdir(); self.bank.mkdir()
        self.code_hash = 'a' * 64
        self.identity = dict(dim=8, source_head='hard', pool_mode='hard', read_mode='hard',
                             front_sha256='b' * 64, readout_sha256='c' * 64)
        train = ['event%d_tile%04d' % (index % 2, index) for index in range(2918)]
        names = train[:2] if smoke else train
        names = names if bank_names is None else names[:bank_names]
        mapping = [index % 2 for index in range(len(names))]
        self.bundle = dict(train=train, device=torch.device('cpu'))
        self.args = SimpleNamespace(memory_run=str(self.run), bank=str(self.bank),
                                    source_head='hard', pool_mode='hard', read_mode='hard',
                                    smoke=smoke, topk=999, seed=999)
        bank_base = dict(format=p.BANK_FORMAT, split='train', test_accessed=False,
                         smoke=smoke, identity=self.identity, image_names=names,
                         event_names=['event0', 'event1'], image_event_ids=mapping, per_bucket=64,
                         tokens=torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64,
                         distribution=torch.eye(4).repeat(2, 1), labelled_mass=torch.ones(8),
                         hard_count=torch.ones(8, dtype=torch.long), purity=torch.ones(8),
                         entropy=torch.zeros(8), images=torch.arange(8) % 2,
                         events=torch.arange(8) % 2, regions=torch.arange(8))
        for policy in ('all', 'hard'):
            torch.save(dict(bank_base, policy=policy), self.bank / ('bank_' + policy + '.pt'))
        bank_manifest = dict(format=p.BANK_FORMAT, complete=True, split='train',
                             test_accessed=False, smoke=smoke, identity=self.identity,
                             source_hash=self.code_hash, dim=8, image_count=len(names), per_bucket=64,
                             image_names=names, event_names=bank_base['event_names'],
                             all_regions=8, hard_regions=8, bank_limit_per_policy=512,
                             all_sha256=p.common.digest(self.bank / 'bank_all.pt'),
                             hard_sha256=p.common.digest(self.bank / 'bank_hard.pt'))
        _write_json(self.bank / 'manifest.json', bank_manifest)
        self.bank_hashes = {str(self.bank / name): p.common.digest(self.bank / name)
                            for name in p.BANK_FILES}
        step = 20 if smoke else 5000
        self.expected = p.common.metrics(torch.eye(4, dtype=torch.long) * 1000000)
        self.manifest = dict(format=p.HEAD_FORMAT, complete=True, smoke=smoke,
                             test_accessed=False, identity=self.identity, code_hash=self.code_hash,
                             bank_format=p.BANK_FORMAT, bank_file_sha256=self.bank_hashes,
                             all_bank_regions=8, hard_bank_regions=8, bank_limit_per_policy=512,
                             frozen_front=True, frozen_building=True, frozen_q=True,
                             frozen_embedding=True, frozen_source_classifier=True,
                             arguments=dict(source_head='hard', pool_mode='hard', read_mode='hard',
                                            smoke=smoke, steps=step, seed=3407, topk=8,
                                            query_chunk=256, allow_same_event=False))
        self.summary = dict(complete=True, phase='complete', smoke=smoke, test_accessed=False,
                            final_step=step, optimizer_steps=step, frozen_source_verified=True,
                            bank_immutable_verified=True, v2_code_immutable_verified=True,
                            baseline_reproduction_pass=None if smoke else True,
                            final_metrics={name: copy.deepcopy(self.expected) for name in p.HEADS.values()})
        _write_json(self.run / 'manifest.json', self.manifest)
        _write_json(self.run / 'summary.json', self.summary)
        self.final_row = dict(step=step, image_count=2 if smoke else 450, test_accessed=False,
                              metrics=copy.deepcopy(self.summary['final_metrics']))
        self.write_history([self.final_row])
        for name in p.HEADS.values():
            head = ResidualSupportHead(8)
            with torch.no_grad():
                head.ffn[-1].bias.fill_(.05)
            checkpoint = dict(format=p.HEAD_FORMAT, name=name, step=step, head=head.state_dict(),
                              metrics=copy.deepcopy(self.expected), source_identity=self.identity,
                              code_hash=self.code_hash, bank_file_sha256=self.bank_hashes)
            torch.save(checkpoint, self.run / (name + '_last.pth'))

    def write_history(self, rows):
        (self.run / 'validation.jsonl').write_text(
            ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')

    def patch_checkpoint(self, **values):
        path = self.run / 'balanced_hard_last.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint.update(values)
        torch.save(checkpoint, path)

    def refresh_bank_provenance(self):
        self.bank_hashes = {str(self.bank / name): p.common.digest(self.bank / name)
                            for name in p.BANK_FILES}
        self.manifest['bank_file_sha256'] = self.bank_hashes
        _write_json(self.run / 'manifest.json', self.manifest)
        for name in p.HEADS.values():
            path = self.run / (name + '_last.pth')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint['bank_file_sha256'] = self.bank_hashes
            torch.save(checkpoint, path)

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(p.common, 'source_identity', return_value=self.identity))
        stack.enter_context(mock.patch.object(p.common, 'experiment_hash', return_value=self.code_hash))
        stack.enter_context(mock.patch.object(p.common, 'assert_v2_immutable'))
        return stack

    def load(self):
        return p.load_trained(self.args, self.bundle)


class LoadingContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = TrainedFixture(Path(self.temporary.name))
        self.stack = self.fixture.mocks()
        self.addCleanup(self.stack.close)

    def test_final_heads_frozen_and_original_settings_preserved(self):
        state = self.fixture.load()
        self.assertEqual(set(state['heads']), {'nomemory', 'normal'})
        self.assertEqual(state['settings'], dict(seed=3407, topk=8, query_chunk=256, allow_same_event=False))
        self.assertEqual(set(state['states']), set(state['heads']))
        self.assertEqual(len(state['files']), 8)
        for head in state['heads'].values():
            self.assertFalse(head.training)
            self.assertTrue(all(not value.requires_grad and value.grad is None for value in head.parameters()))
        p.assert_trained_unchanged(state, self.fixture.bundle)

    def test_rejects_different_final_step_and_name(self):
        for change in ({'step': 4500}, {'step': True}, {'name': 'global_hard'}):
            with self.subTest(change=change):
                self.fixture.patch_checkpoint(step=5000, name='balanced_hard')
                self.fixture.patch_checkpoint(**change)
                with self.assertRaisesRegex(ValueError, 'same-final-step'):
                    self.fixture.load()

    def test_rejects_checkpoint_identity_or_code_or_bank_swap(self):
        defaults = dict(source_identity=self.fixture.identity, code_hash=self.fixture.code_hash,
                        bank_file_sha256=self.fixture.bank_hashes)
        changes = [dict(source_identity=dict(self.fixture.identity, dim=12)),
                   dict(code_hash='d' * 64), dict(bank_file_sha256={})]
        for change in changes:
            with self.subTest(change=change):
                self.fixture.patch_checkpoint(**defaults)
                self.fixture.patch_checkpoint(**change)
                with self.assertRaises(ValueError):
                    self.fixture.load()

    def test_rejects_incomplete_and_test_accessed_and_budget_mismatch(self):
        original = copy.deepcopy(self.fixture.summary)
        for change in (dict(complete=False), dict(test_accessed=True), dict(optimizer_steps=4500),
                       dict(baseline_reproduction_pass=False), dict(smoke=True)):
            with self.subTest(change=change):
                _write_json(self.fixture.run / 'summary.json', dict(original, **change))
                with self.assertRaises(ValueError):
                    self.fixture.load()

    def test_rejects_history_duplicates_or_wrong_final_metrics(self):
        row = self.fixture.final_row
        self.fixture.write_history([row, row])
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            self.fixture.load()
        different = copy.deepcopy(row)
        different['metrics']['nomemory']['valid_miou'] -= .01
        self.fixture.write_history([different])
        with self.assertRaisesRegex(ValueError, 'history/summary'):
            self.fixture.load()

    def test_rejects_nonfinite_missing_and_wrong_dtype_head_tensors(self):
        original = ResidualSupportHead(8).state_dict()
        variants = []
        nonfinite = copy.deepcopy(original); nonfinite['labels.weight'][0, 0] = float('nan')
        variants.append(nonfinite)
        missing = copy.deepcopy(original); missing.pop('labels.weight')
        variants.append(missing)
        wrong_dtype = copy.deepcopy(original); wrong_dtype['labels.weight'] = wrong_dtype['labels.weight'].double()
        variants.append(wrong_dtype)
        for weights in variants:
            with self.subTest(keys=list(weights)):
                self.fixture.patch_checkpoint(head=weights)
                with self.assertRaises(ValueError):
                    self.fixture.load()

    def test_rejects_bank_manifest_code_even_with_matching_file_hashes(self):
        path = self.fixture.bank / 'manifest.json'
        value = json.loads(path.read_text())
        value['source_hash'] = 'd' * 64
        _write_json(path, value)
        self.fixture.refresh_bank_provenance()
        with self.assertRaisesRegex(ValueError, 'Bank extraction code'):
            self.fixture.load()

    def test_rejects_ambiguous_bank_hash_names(self):
        record = dict(self.fixture.bank_hashes)
        record.pop(str(self.fixture.bank / 'bank_all.pt'))
        record['somewhere/manifest.json'] = 'a' * 64
        self.fixture.manifest['bank_file_sha256'] = record
        _write_json(self.fixture.run / 'manifest.json', self.fixture.manifest)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.fixture.load()

    def test_detects_head_mutation_and_training_flag(self):
        state = self.fixture.load()
        state['heads']['normal'].train()
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            p.assert_trained_unchanged(state, self.fixture.bundle)
        state['heads']['normal'].eval()
        with torch.no_grad():
            state['heads']['normal'].labels.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'state changed'):
            p.assert_trained_unchanged(state, self.fixture.bundle)

    def test_detects_loaded_bank_mutation(self):
        state = self.fixture.load()
        state['banks']['hard']['tokens'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Loaded Bank'):
            p.assert_trained_unchanged(state, self.fixture.bundle)

    def test_detects_source_file_and_old_implementation_changes(self):
        state = self.fixture.load()
        with mock.patch.object(p.common, 'assert_v2_immutable', side_effect=RuntimeError('code changed')):
            with self.assertRaisesRegex(ValueError, 'code changed'):
                p.assert_trained_unchanged(state, self.fixture.bundle)
        with (self.fixture.run / 'summary.json').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'source file changed'):
            p.assert_trained_unchanged(state, self.fixture.bundle)

    def test_rejects_duplicate_json_keys(self):
        (self.fixture.run / 'summary.json').write_text('{"complete":true,"complete":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            self.fixture.load()

    def test_corrupt_checkpoint_is_reported_as_value_error(self):
        (self.fixture.run / 'balanced_hard_last.pth').write_bytes(b'not a checkpoint')
        with self.assertRaises(ValueError):
            self.fixture.load()


class BankCoverageContracts(unittest.TestCase):
    def test_formal_train_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TrainedFixture(Path(directory), bank_names=2)
            with fixture.mocks(), self.assertRaisesRegex(ValueError, 'complete unique Train2918'):
                fixture.load()

    def test_audit_smoke_uses_formal_sources_without_full_val_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TrainedFixture(Path(directory))
            fixture.args.smoke = True
            with fixture.mocks():
                state = fixture.load()
                result = p.verify_trained_metrics({}, state, True)
                self.assertIsNone(result['passed'])
                self.assertEqual(result['status'], 'smoke_not_full_val')

    def test_audit_smoke_rejects_smoke_trained_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TrainedFixture(Path(directory), smoke=True)
            with fixture.mocks(), self.assertRaisesRegex(ValueError, 'formal trained heads'):
                fixture.load()

    def test_audit_smoke_rejects_smoke_banks_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TrainedFixture(Path(directory))
            fixture.args.smoke = True
            manifest = json.loads((fixture.bank / 'manifest.json').read_text())
            for policy in ('all', 'hard'):
                path = fixture.bank / ('bank_' + policy + '.pt')
                bank = torch.load(path, map_location='cpu', weights_only=False)
                bank['smoke'] = True
                torch.save(bank, path)
                manifest[policy + '_sha256'] = p.common.digest(path)
            manifest['smoke'] = True
            _write_json(fixture.bank / 'manifest.json', manifest)
            fixture.refresh_bank_provenance()
            with fixture.mocks(), self.assertRaisesRegex(ValueError, 'smoke/formal mismatch'):
                fixture.load()


class FixedMetricGate(unittest.TestCase):
    def setUp(self):
        self.expected = p.common.metrics(torch.eye(4, dtype=torch.long) * 1000000)
        self.state = dict(summary=dict(final_metrics={name: copy.deepcopy(self.expected)
                                                     for name in p.HEADS.values()}))
        self.actual = {name: copy.deepcopy(self.expected) for name in p.HEADS}

    def test_all_metrics_are_checked_with_fixed_tolerance(self):
        report = p.verify_trained_metrics(self.actual, self.state, False)
        self.assertTrue(report['passed'])
        self.assertEqual(report['tolerance'], 1e-5)
        fields = {row['metric'] for row in report['checks']}
        self.assertTrue(set(p.FLOAT_FIELDS + p.EXACT_FIELDS).issubset(fields))
        self.actual['normal']['f1'][2] -= 2e-5
        self.assertFalse(p.verify_trained_metrics(self.actual, self.state, False)['passed'])
        self.assertEqual(p.TOLERANCE, 1e-5)

    def test_confusion_changes_within_metric_tolerance_are_diagnostic(self):
        confusion = torch.eye(4, dtype=torch.long) * 1000000
        confusion[0, 0] -= 1; confusion[0, 1] += 1
        self.actual['normal'] = p.common.metrics(confusion)
        report = p.verify_trained_metrics(self.actual, self.state, False)
        self.assertTrue(report['passed'])
        differences = [row for row in report['diagnostic_differences'] if not row['equal']]
        self.assertTrue(differences)
        self.assertTrue(all(row['gate'] is False for row in differences))

    def test_count_and_gt_support_are_exact(self):
        confusion = torch.eye(4, dtype=torch.long) * 1000000
        confusion[0, 0] += 1
        self.actual['normal'] = p.common.metrics(confusion)
        report = p.verify_trained_metrics(self.actual, self.state, False)
        self.assertFalse(report['passed'])
        failures = {row['metric'] for row in report['checks'] if not row['passed']}
        self.assertEqual(failures, {'count', 'support'})

    def test_absent_class_missing_nonfinite_and_bool_metrics_fail(self):
        for value in (None, float('nan'), True):
            with self.subTest(value=value):
                actual = copy.deepcopy(self.actual)
                actual['normal']['iou'][1] = value
                report = p.verify_trained_metrics(actual, self.state, False)
                self.assertFalse(report['passed'])
                json.dumps(report, allow_nan=False)
        missing = copy.deepcopy(self.actual); missing['normal'].pop('precision')
        self.assertFalse(p.verify_trained_metrics(missing, self.state, False)['passed'])
        self.assertFalse(p.verify_trained_metrics({}, self.state, False)['passed'])


if __name__ == '__main__':
    unittest.main()
