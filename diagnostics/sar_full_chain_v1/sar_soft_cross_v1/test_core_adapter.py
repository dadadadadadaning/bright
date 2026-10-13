"""CPU contracts for the scoped bridge; no model training or CUDA validation."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import core_adapter as adapter


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def original_function(package, filename, name):
    """Use the installed original Python gate, without importing its models."""
    candidates = [Path(os.environ.get('BRIGHT_REPO', '/root/shared-nvme/BRIGHT-master')),
                  Path('/workspace/bright'), Path.cwd()]
    relative = Path('bda_benchmark/experiments') / package / filename
    path = None
    for root in candidates:
        try:
            if (root / relative).is_file():
                path = root / relative
                break
        except OSError:
            continue
    if path is None:
        raise unittest.SkipTest('Installed BRIGHT sources needed for unchanged-gate contracts')
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class FakeFront:
    def __init__(self):
        for name in ('base', 'adapter', 'embedding', 'region_classifier'):
            setattr(self, name, SimpleNamespace(signature='same_' + name))
        self.frozen = False
        self.evaluation = False

    def requires_grad_(self, value):
        self.frozen = value is False
        return self

    def eval(self):
        self.evaluation = True
        return self


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(random.setstate, random.getstate())
        self.external = self.root / 'external_source'
        self.external.mkdir()
        (self.external / 'core_adapter.py').write_text('# bound synthetic external bridge source\n')
        source_location = patch.object(adapter, 'HERE', self.external)
        source_location.start()
        self.addCleanup(source_location.stop)
        self.q_path = self.root / 'q' / 'last.pth'
        self.q_path.parent.mkdir()
        self.q_path.write_bytes(b'original Q fixture bytes; model loading is mocked')
        self.protocol = self.root / 'protocol.json'
        write(self.protocol, {'synthetic': 'E0 binding'})
        self.protocol_sha = adapter.digest(self.protocol)
        self.front = FakeFront()
        self.model = FakeFront()
        self.bundle = dict(front=self.front, device='cpu',
            args=SimpleNamespace(a1_checkpoint=str(self.root / 'a1.pth'), seed=3407),
            protocol_sha256=self.protocol_sha, hashes={'a1_sha256': 'a1hash', 'front_sha256': 'fronthash'},
            input_paths={}, inputs_before={})
        self.q_arguments = dict(steps=2000, eval_every=2000, batch=2, accum=8, workers=4,
                               lr=1e-4, aux_weight=.1, cue_gain=.1, amp='bf16')
        self.payload = dict(format='SRAC_Q_FACTORIAL_V1', run_complete=True, test_accessed=False,
            immutable={'source': True, 'models': True}, a1_unchanged=True, downstream_unchanged=True,
            protocol_sha256=self.protocol_sha, a1_sha256='a1hash', front_sha256='fronthash',
            smoke=False, completed_steps=2000, step=2000, arguments=self.q_arguments,
            arm='baseline', source_hash='original_q_source', inherited_source_hash='original_inherited_source',
            cue_gain=.1)
        self.events = []
        def load(*args, **kwargs):
            self.events.append('load')
            # Model constructors for different Q arms may consume different RNG amounts.
            for _ in range(3 if self.payload['arm'] == 'baseline' else 17):
                random.random()
            return self.model, self.payload
        checked = original_function('damageformer_srac_q_factorial_v1', 'evaluate_q.py', '_checked_checkpoint')
        self.checked = Mock(wraps=checked)
        self.loader = Mock(side_effect=load)
        self.seed = Mock(side_effect=lambda value: (self.events.append(('seed', value)), random.seed(value)))
        self.q_variants = SimpleNamespace(load_q_checkpoint=self.loader, q_source_hash=lambda: 'original_q_source')
        self.q_eval = SimpleNamespace(_checked_checkpoint=self.checked,
                                     _same_base=lambda a, b: a.signature == b.signature)
        self.common = SimpleNamespace(seed_everything=self.seed)
        self.train_front = SimpleNamespace(module_fingerprint=lambda part:
                                           hashlib.sha256(part.signature.encode()).hexdigest())
        self.old_hash = lambda: 'original_route_hash'
        self.old_loader = lambda *a, **kw: ('unchanged', {})
        self.core = SimpleNamespace(ARMS=adapter.ARMS, FORMAT='SRAC_BUILDING_ROUTE_V1',
                                   source_hash=self.old_hash, load_route_source=self.old_loader)
        original_audit = original_function('damageformer_srac_building_route_v1', 'train.py',
                                           'assert_completed_audit')
        self.audit = Mock(wraps=original_audit)
        self.consumers = {name: SimpleNamespace(FORMAT=self.core.FORMAT,
                            source_hash=self.old_hash, load_route_source=self.old_loader)
                          for name in ('train', 'evaluate', 'report')}
        self.consumers['train'].assert_completed_audit = self.audit
        self.modules = {adapter.ROUTE_MODULE + '.core': self.core,
                        adapter.Q_MODULE + '.q_variants': self.q_variants,
                        adapter.Q_MODULE + '.evaluate_q': self.q_eval,
                        'bda_benchmark.experiments.damageformer_srac_transformer_v1.common': self.common,
                        'bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front': self.train_front}
        self.modules.update({adapter.ROUTE_MODULE + '.' + name: module
                             for name, module in self.consumers.items()})
        importer = patch.object(adapter.importlib, 'import_module', side_effect=lambda name: self.modules[name])
        importer.start()
        self.addCleanup(importer.stop)
        self.assertFalse(adapter._ACTIVE)
        self.addCleanup(setattr, adapter, '_ACTIVE', False)

    def metadata(self):
        return dict(kind='completed_baseline_q', arm='baseline', checkpoint=str(self.q_path),
                    sha256=adapter.digest(self.q_path), steps=2000, protocol_sha256=self.protocol_sha,
                    route_seed=3438, train_source_hash='original_q_source',
                    inherited_source_hash='original_inherited_source', cue_gain=.1,
                    training_hyperparameters=self.q_arguments,
                    initial_readout_fingerprints={name: self.train_front.module_fingerprint(getattr(self.model, name))
                                                 for name in ('embedding', 'region_classifier')})

    def test_baseline_and_sar_use_original_formal_gate_even_for_route_smoke(self):
        before = self.q_path.read_bytes()
        for arm in ('baseline', 'sar_cue'):
            with self.subTest(arm=arm):
                self.payload['arm'] = arm
                model, meta = adapter.load_route_source(self.bundle, self.q_path, smoke=True)
                self.assertIs(model, self.model)
                self.loader.assert_called_with(self.q_path, 'cpu',
                    a1_override=self.bundle['args'].a1_checkpoint, allow_smoke=False)
                self.checked.assert_called_with(self.payload, self.bundle, adapter.digest(self.q_path), False)
                self.assertEqual(meta['arm'], arm)
                self.assertEqual(meta['kind'], 'completed_' + arm + '_q')
                self.assertEqual(meta['steps'], 2000)
                self.assertEqual(meta['route_seed'], 3438)
                self.assertEqual(set(meta['initial_readout_fingerprints']), {'embedding', 'region_classifier'})
                self.assertTrue(all(len(x) == 64 for x in meta['initial_readout_fingerprints'].values()))
                self.assertTrue(model.frozen and model.evaluation)
                self.assertEqual(self.bundle['inputs_before']['q_checkpoint'], adapter.digest(self.q_path))
        self.assertEqual(before, self.q_path.read_bytes())

    def test_loader_failure_source_hash_gate_is_propagated(self):
        self.loader.side_effect = ValueError('Original Q source hash gate rejected revision')
        with self.assertRaisesRegex(ValueError, 'Original Q source hash'):
            adapter.load_route_source(self.bundle, self.q_path)
        self.checked.assert_not_called()
        self.seed.assert_not_called()
        self.assertEqual(self.bundle['input_paths'], {})

    def test_failed_smoke_wrong_protocol_and_downstream_certificates_rejected(self):
        original = copy.deepcopy(self.payload)
        cases = ({'run_complete': False}, {'smoke': True}, {'completed_steps': 2, 'step': 2},
                 {'test_accessed': True}, {'protocol_sha256': 'other_e0'}, {'a1_sha256': 'other_a1'},
                 {'downstream_unchanged': False}, {'immutable': {'source': False}})
        for change in cases:
            with self.subTest(change=change):
                self.payload = dict(original, **change)
                with self.assertRaises(ValueError):
                    adapter.load_route_source(self.bundle, self.q_path, smoke=True)
        self.seed.assert_not_called()
        self.assertEqual(self.bundle['input_paths'], {})

    def test_e0_missing_checkpoint_and_wrong_q_arm_refused(self):
        with self.assertRaisesRegex(ValueError, 'completed baseline'):
            adapter.load_route_source(self.bundle)
        self.payload['arm'] = 'building_reconstruction'
        with self.assertRaisesRegex(ValueError, 'baseline/sar_cue'):
            adapter.load_route_source(self.bundle, self.q_path)
        self.assertEqual(self.bundle['input_paths'], {})

    def test_frozen_component_weights_must_equal_original_front(self):
        for name in ('base', 'adapter', 'embedding', 'region_classifier'):
            with self.subTest(name=name):
                part = getattr(self.model, name)
                old = part.signature
                part.signature = 'different weights'
                with self.assertRaisesRegex(ValueError, name):
                    adapter.load_route_source(self.bundle, self.q_path)
                part.signature = old
        self.seed.assert_not_called()

    def test_missing_q_settings_and_invalid_seed_refused(self):
        del self.q_arguments['batch']
        with self.assertRaisesRegex(ValueError, 'training configuration'):
            adapter.load_route_source(self.bundle, self.q_path)
        self.q_arguments['batch'] = 2
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                self.bundle['args'].seed = value
                with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                    adapter.load_route_source(self.bundle, self.q_path)

    def test_common_rng_reset_occurs_after_each_q_constructor(self):
        results = []
        for arm in ('baseline', 'sar_cue'):
            self.payload['arm'] = arm
            random.seed(129)
            adapter.load_route_source(self.bundle, self.q_path)
            results.append(random.random())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], random.Random(3438).random())
        self.assertEqual(self.events, ['load', ('seed', 3438), 'load', ('seed', 3438)])

    def test_checkpoint_change_during_load_is_rejected(self):
        def changed(*args, **kwargs):
            self.q_path.write_bytes(b'changed during mocked model construction')
            return self.model, self.payload
        self.loader.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'changed while loading'):
            adapter.load_route_source(self.bundle, self.q_path)
        self.assertEqual(self.bundle['inputs_before'], {})

    def assert_restored(self):
        self.assertFalse(adapter._ACTIVE)
        self.assertIs(self.core.source_hash, self.old_hash)
        self.assertIs(self.core.load_route_source, self.old_loader)
        self.assertEqual(self.core.FORMAT, 'SRAC_BUILDING_ROUTE_V1')
        for module in self.consumers.values():
            self.assertEqual(module.FORMAT, self.core.FORMAT)
            self.assertIs(module.source_hash, self.old_hash)
            self.assertIs(module.load_route_source, self.old_loader)

    def test_scoped_consumers_use_bridge_but_original_core_and_q_are_untouched(self):
        checked, loader = self.q_eval._checked_checkpoint, self.q_variants.load_q_checkpoint
        before = self.q_path.read_bytes()
        with adapter.adapted_modules() as modules:
            self.assertEqual(set(modules), {'train', 'evaluate', 'report'})
            for module in modules.values():
                self.assertEqual(module.FORMAT, adapter.FORMAT)
                self.assertIs(module.source_hash, adapter.source_hash)
                self.assertIs(module.load_route_source, adapter.load_route_source)
            self.assertIs(self.core.source_hash, self.old_hash)
            self.assertIs(self.core.load_route_source, self.old_loader)
            self.assertIs(self.q_eval._checked_checkpoint, checked)
            self.assertIs(self.q_variants.load_q_checkpoint, loader)
        self.assert_restored()
        self.assertEqual(before, self.q_path.read_bytes())

    def test_context_failure_and_nested_rejection_restore_aliases(self):
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            with adapter.adapted_modules():
                with self.assertRaisesRegex(RuntimeError, 'Nested'):
                    with adapter.adapted_modules():
                        self.fail('Nested adapter accepted')
                raise RuntimeError('worker failed')
        self.assert_restored()

    def test_partial_context_entry_restores_previously_patched_modules(self):
        self.consumers['evaluate'].FORMAT = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'already adapted'):
            with adapter.adapted_modules():
                self.fail('Wrong consumer schema accepted')
        self.assertFalse(adapter._ACTIVE)
        self.assertEqual(self.consumers['train'].FORMAT, self.core.FORMAT)
        self.assertIs(self.consumers['train'].source_hash, self.old_hash)
        self.assertEqual(self.consumers['evaluate'].FORMAT, 'unexpected')

    def test_source_hash_binds_original_q_route_and_external_code_excludes_tests(self):
        external = self.root / 'external'
        external.mkdir()
        code = external / 'core_adapter.py'
        code.write_text('# original external implementation\n')
        test = external / 'test_core_adapter.py'
        test.write_text('# original test\n')
        with patch.object(adapter, 'HERE', external):
            first = adapter.source_hash()
            test.write_text('# changed test\n')
            self.assertEqual(first, adapter.source_hash())
            code.write_text('# changed implementation\n')
            second = adapter.source_hash()
            self.assertNotEqual(first, second)
            self.q_variants.q_source_hash = lambda: 'changed_q_source'
            third = adapter.source_hash()
            self.assertNotEqual(second, third)
            self.core.source_hash = adapter.source_hash
            with self.assertRaisesRegex(RuntimeError, 'recursive'):
                adapter.source_hash()

    def make_fit(self, smoke=False):
        fit = self.root / ('smoke_fit' if smoke else 'formal_fit')
        fit.mkdir()
        steps = 2 if smoke else 2000
        q = self.metadata()
        arguments = dict(steps=steps, smoke=smoke, threshold=.5, soft_floor=0., lr=1e-4,
                         eval_every=steps, q_checkpoint=str(self.q_path), protocol=str(self.protocol), output=str(fit))
        shared = dict(format=adapter.FORMAT, protocol_sha256=self.protocol_sha,
                      source_sha256=adapter.source_hash(), smoke=smoke, test_accessed=False, q_source=q)
        params = {arm: 19 for arm in adapter.ARMS}
        manifest = dict(shared, arguments=arguments, trainable_parameters=params)
        summary = dict(shared, complete=True, training=False, full_validation_evidence=not smoke,
                       steps=steps, updates_per_arm={arm: steps for arm in adapter.ARMS},
                       immutable={'weights': True, 'files': True}, checkpoint='heads_last.pth',
                       threshold=.5, soft_floor=0., trainable_parameters=params)
        heads = {arm: {'embedding': {'weight': torch.tensor([1.])},
                       'region_classifier': {'weight': torch.tensor([2.])}} for arm in adapter.ARMS}
        state = dict(shared, complete=True, completed_steps=steps, threshold=.5, soft_floor=0.,
                     arguments=arguments, heads=heads, optimizer_states={arm: {} for arm in adapter.ARMS})
        torch.save(state, fit / 'heads_last.pth')
        summary['checkpoint_sha256'] = adapter.digest(fit / 'heads_last.pth')
        stream = hashlib.sha256()
        lines = []
        for step in range(1, steps + 1):
            image = 'synthetic_image_' + str(step % 3)
            lines.append(json.dumps({'step': step, 'image': image}))
            stream.update((str(step) + '\t' + image + '\n').encode())
        (fit / 'training.jsonl').write_text('\n'.join(lines) + '\n')
        summary['training_journal_sha256'] = adapter.digest(fit / 'training.jsonl')
        summary['train_stream_sha256'] = stream.hexdigest()
        for initial in (True, False):
            folder = fit / ('initial_val' if initial else 'final_val')
            report = dict(format=adapter.FORMAT + '_EVAL', complete=True, smoke=smoke,
                          test_accessed=False, image_count=2 if smoke else 450,
                          full_validation_evidence=not smoke, protocol_sha256=self.protocol_sha,
                          source_sha256=shared['source_sha256'], q_source=q, native_a1_replay_ok=True,
                          immutable={'files': True}, initial=initial, reproduction={'passed': True},
                          e0_metric_comparison={'passed': True}, q_a1_reproduction={'passed': True},
                          metrics={arm: {'valid_miou': .61} for arm in adapter.ARMS})
            write(folder / 'result.json', report)
            audit_summary = dict(format=adapter.FORMAT + '_EVAL', complete=True, smoke=smoke,
                                 test_accessed=False, image_count=2 if smoke else 450,
                                 full_validation_evidence=not smoke, result_file='result.json',
                                 native_a1_replay_ok=True, immutable={'files': True},
                                 reproduction_passed=True, e0_protocol_comparison_passed=True,
                                 q_a1_reproduction_passed=True,
                                 result_sha256=adapter.digest(folder / 'result.json'))
            write(folder / 'summary.json', audit_summary)
            summary['initial' if initial else 'final'] = {k: report[k] for k in (
                'complete', 'full_validation_evidence', 'reproduction', 'e0_metric_comparison',
                'q_a1_reproduction', 'metrics')}
        write(fit / 'manifest.json', manifest)
        write(fit / 'summary.json', summary)
        return fit

    def test_complete_formal_and_smoke_fit_prechecks_are_separate_read_only(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                fit = self.make_fit(smoke)
                before = {str(p): adapter.digest(p) for p in fit.rglob('*') if p.is_file()}
                result = adapter.precheck_fit(fit, self.protocol, smoke=smoke)
                self.assertEqual(result['summary']['steps'], 2 if smoke else 2000)
                self.assertEqual(result['summary']['full_validation_evidence'], not smoke)
                self.assertEqual(before, {str(p): adapter.digest(p) for p in fit.rglob('*') if p.is_file()})
        self.assertEqual(self.audit.call_count, 4)
        with self.assertRaises(ValueError):
            adapter.precheck_fit(self.root / 'smoke_fit', self.protocol, smoke=False)

    def test_partial_failed_wrong_source_or_protocol_fit_is_rejected(self):
        fit = self.make_fit()
        summary = adapter.read_json(fit / 'summary.json')
        for changes in ({'complete': False}, {'full_validation_evidence': False},
                        {'updates_per_arm': {a: 1999 for a in adapter.ARMS}},
                        {'smoke': True}, {'source_sha256': 'wrong_source'},
                        {'protocol_sha256': 'wrong_e0'}, {'immutable': {'weights': False}}):
            with self.subTest(changes=changes):
                write(fit / 'summary.json', dict(summary, **changes))
                with self.assertRaises(ValueError):
                    adapter.precheck_fit(fit, self.protocol)
        write(fit / 'summary.json', summary)

    def test_fit_checkpoint_metadata_or_nonfinite_head_state_refused(self):
        fit = self.make_fit()
        original = torch.load(fit / 'heads_last.pth', map_location='cpu', weights_only=False)
        summary = adapter.read_json(fit / 'summary.json')
        for case in ('complete', 'steps', 'heads', 'nonfinite'):
            with self.subTest(case=case):
                state = copy.deepcopy(original)
                if case == 'complete':
                    state['complete'] = False
                elif case == 'steps':
                    state['completed_steps'] = 1999
                elif case == 'heads':
                    del state['heads']['soft']
                else:
                    state['heads']['soft']['embedding']['weight'] = torch.tensor([float('nan')])
                torch.save(state, fit / 'heads_last.pth')
                summary['checkpoint_sha256'] = adapter.digest(fit / 'heads_last.pth')
                write(fit / 'summary.json', summary)
                with self.assertRaises(ValueError):
                    adapter.precheck_fit(fit, self.protocol)

    def test_precheck_rejects_modified_q_checkpoint_and_training_journal(self):
        fit = self.make_fit()
        old_q = self.q_path.read_bytes()
        self.q_path.write_bytes(b'changed source Q after route training')
        with self.assertRaisesRegex(ValueError, 'Original Q checkpoint hash changed'):
            adapter.precheck_fit(fit, self.protocol)
        self.q_path.write_bytes(old_q)
        with (fit / 'training.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'Training journal hash changed'):
            adapter.precheck_fit(fit, self.protocol)

    def test_incomplete_or_wrong_role_saved_val_cannot_certify_fit(self):
        fit = self.make_fit()
        summary_path = fit / 'final_val' / 'summary.json'
        audit_summary = adapter.read_json(summary_path)
        write(summary_path, dict(audit_summary, image_count=449))
        with self.assertRaisesRegex(ValueError, 'audit is incomplete'):
            adapter.precheck_fit(fit, self.protocol)
        write(summary_path, audit_summary)
        result_path = fit / 'final_val' / 'result.json'
        result = adapter.read_json(result_path)
        result['initial'] = True
        write(result_path, result)
        audit_summary['result_sha256'] = adapter.digest(result_path)
        write(summary_path, audit_summary)
        with self.assertRaisesRegex(ValueError, 'audit role differs'):
            adapter.precheck_fit(fit, self.protocol)


if __name__ == '__main__':
    unittest.main()
