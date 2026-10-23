import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import replay


def fingerprint(value):
    def encode(v):
        if isinstance(v, torch.Tensor):
            return (str(v.dtype), list(v.shape), v.detach().cpu().tolist())
        if isinstance(v, dict):
            return {str(k): encode(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        package = self.root / 'package'
        package.mkdir()
        (package / 'experiment.py').write_text('IMPORT_IS_STDLIB_ONLY = True\n')
        identity = {'experiment.py': replay.sha(package / 'experiment.py')}
        parent = self.root / 'train_original'
        (parent / 'checkpoints').mkdir(parents=True)
        plan = dict(format=replay.PARENT_FORMAT, variant='original', effective_mode='original', seed=3407,
                    endpoint_step=2500, read_weight=0, extension_files=identity,
                    scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                    warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
                    post500_lr_multiplier=.2, test_accessed=False)
        train = ['train_'+str(i) for i in range(2918)]
        train[157] = replay.TARGET_IMAGE
        manifest = dict(plan=plan, seed=3407, train_names=train, actual_training_names=train,
                        val_names=['val_'+str(i) for i in range(450)])
        ckpt = parent / 'checkpoints/checkpoint_000150.pt'
        ckpt.write_bytes(b'preflight does not deserialize a checkpoint')
        sidecar = dict(step=150, sha256=replay.sha(ckpt), roundtrip_verified=True, variant='original')
        replay.write(ckpt.with_suffix('.pt.json'), sidecar)
        replay.write(parent/'plan.json', plan)
        replay.write(parent/'manifest.json', manifest)
        replay.write(parent/'summary.json', dict(phase='failed', last_complete_step=157,
                     failure_phase='update_158', current_image=replay.TARGET_IMAGE, last_checkpoint=sidecar))
        replay.write(parent/'first_failure.json', dict(phase='update_158', image=replay.TARGET_IMAGE,
                                                     last_complete_step=157))
        rows = [dict(step=i, optimizer_step=i, epoch=0, epoch_seed=3407, variant='original',
                     image=train[i-1]) for i in range(1,158)]
        (parent/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return package, parent, identity, manifest, rows

    def severity_experiment(self):
        package = Path(os.environ.get('BRIGHT_SEVERITY_PACKAGE',
                                      '/workspace/bright_severity_memory_20261023'))
        path = package / 'experiment.py'
        if not path.is_file():
            self.skipTest('Original update dependency unavailable; set BRIGHT_SEVERITY_PACKAGE: '+str(path))
        spec = importlib.util.spec_from_file_location('_backward158_original_update_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return package, module

    @staticmethod
    def toy_update_fixture():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_projection = torch.nn.Parameter(torch.tensor([.2, -.4, .8, .1]))
                self.label_embedding = torch.nn.Parameter(torch.tensor([.4, .1, -.2, .3]))

            def forward(self, **kwargs):
                value = self.query_projection + .2 * self.label_embedding
                gathered = value.gather(0, torch.tensor([2, 0, 3, 1]))
                return {'probabilities': gathered.sort(-1).values.softmax(0)}

        example = types.SimpleNamespace(
            common={'q': torch.ones(1), 'ids': torch.zeros(1, dtype=torch.long)},
            target=torch.tensor([0, 2]))
        support = {'route_q': torch.ones(1), 'route_ids': torch.zeros(1, dtype=torch.long),
                   'support_ids': torch.zeros(1, dtype=torch.long)}
        factory = types.SimpleNamespace(prepare=lambda *args: {'d': support})
        runtime = types.SimpleNamespace(loss=lambda p, t: -p.gather(0, t).log().mean())
        state = types.SimpleNamespace(lr_at_step=lambda step, maximum, **kwargs: maximum*step/500)
        return Model(), example, factory, runtime, state

    def compare_successful_updates(self, exp, model, example, factory, runtime, state, slots=None):
        observed = copy.deepcopy(model)
        baseline_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.005)
        observed_optimizer = torch.optim.AdamW(observed.parameters(), lr=1e-4, weight_decay=.005)
        status = replay.initial_status(self.root)
        for step in (1, 2):
            baseline = exp.update(model, baseline_optimizer, example, runtime, factory, state, slots,
                                  step=step, read_weight=0, multiplier=.2)
            result = replay.observed_update(exp, observed, observed_optimizer, example, runtime,
                                            factory, state, slots, step=step, output=self.root,
                                            status=status, multiplier=.2)
            self.assertEqual(fingerprint(model.state_dict()), fingerprint(observed.state_dict()))
            self.assertEqual(fingerprint(baseline_optimizer.state_dict()),
                             fingerprint(observed_optimizer.state_dict()))
            for key in ('original_loss', 'training_loss', 'lr', 'optimizer_step',
                        'grad_norm_preclip', 'label_grad_abs_sum', 'reader_grad_abs_sum'):
                self.assertEqual(baseline[key], result[key], key)
            report = result['observation_summary']
            self.assertTrue(report['complete'])
            self.assertGreater(report['observed_index_nodes'], 0)
            detailed = replay.read(self.root / f'graph_summary_{step:06d}.json')
            self.assertTrue(all(n['prehook_calls'] == 1 for n in detailed['nodes']))
        self.assertEqual(status['optimizer_updates_returned'], 2)

    def test_observed_update_matches_original_gather_sort_model_and_adamw(self):
        _, exp = self.severity_experiment()
        model, example, factory, runtime, state = self.toy_update_fixture()
        self.compare_successful_updates(exp, model, example, factory, runtime, state)

    def test_incomplete_observer_coverage_never_updates_optimizer(self):
        model, example, factory, runtime, state = self.toy_update_fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.005)
        before_model = fingerprint(model.state_dict())
        status = replay.initial_status(self.root)
        with patch('graph_observer.BackwardIndexObserver.finish',
                   return_value={'complete': False, 'observed_index_nodes': 3}), \
                patch.object(optimizer, 'step', wraps=optimizer.step) as step_call, \
                self.assertRaisesRegex(ValueError, 'observation coverage incomplete'):
            replay.observed_update(None, model, optimizer, example, runtime, factory, state, None,
                                   step=1, output=self.root, status=status, multiplier=.2)
        step_call.assert_not_called()
        self.assertEqual(before_model, fingerprint(model.state_dict()))
        # LR assignment precedes the coverage check but AdamW state/weights do not change.
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(status['optimizer_updates_returned'], 0)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue((self.root/'graph_failure_000001.json').is_file())

    def test_actual_original_classifier_observed_update_matches_original(self):
        package, exp = self.severity_experiment()
        fixture_path = package / 'test_severity_adapter.py'
        adapter_path = package / 'severity_adapter.py'
        reference = Path(os.environ.get('BRIGHT_REFERENCE_MODEL',
            '/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2/deps/original/model.py'))
        missing = [str(path) for path in (fixture_path, adapter_path, reference) if not path.is_file()]
        if missing:
            self.skipTest('Actual original-classifier integration dependencies unavailable: '+', '.join(missing))
        adapter = exp.load_file(adapter_path, 'backward158_test_adapter')
        with patch.dict(sys.modules, {'severity_adapter': adapter}):
            fixtures = exp.load_file(fixture_path, 'backward158_test_fixture')
        base, slots = fixtures.load_reference()
        old = exp.load_file(package / 'old_model_adapter.py', 'backward158_test_old_adapter')
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(73)
        model = adapter.make_classifier(base, old, slots, 'original')(hidden=8, heads=2, chunk_size=2)
        with torch.no_grad():
            model.joint_head[-1].weight.normal_(0, .03)
            model.joint_head[-1].bias.normal_(0, .03)
        inputs = fixtures.fixture(slots)
        common_keys = ('pixel_features', 'region_f', 'region_t', 'q', 'ids', 'base_probability_full')
        common = {key: inputs[key] for key in common_keys}
        support = {key: value for key, value in inputs.items() if key not in common_keys}
        shape = common['base_probability_full'].shape
        target = (torch.arange(shape[-2]*shape[-1]) % 4).reshape(1, shape[-2], shape[-1])
        example = types.SimpleNamespace(common=common, target=target)
        factory = types.SimpleNamespace(prepare=lambda *args: {'d': support})

        def loss(probabilities, labels):
            # Exercise the real sparse-unpool GatherBackward graph and a sorted
            # differentiable loss without importing the dataset runtime.
            selected = probabilities.gather(1, labels[:, None]).squeeze(1)
            nll = -selected.log().mean()
            errors = (1-selected).flatten().sort(descending=True).values
            weights = torch.linspace(.5, 1.5, errors.numel())
            return nll + (errors*weights).mean()

        runtime = types.SimpleNamespace(loss=loss)
        state = types.SimpleNamespace(lr_at_step=lambda step, maximum, **kwargs: maximum*step/500)
        self.compare_successful_updates(exp, model, example, factory, runtime, state, slots)

    def test_parent_preflight_hashes_without_checkpoint_load(self):
        package, parent, identity, _, _ = self.fixture()
        with patch.object(replay, 'IDENTITY', identity), patch.object(torch, 'load', side_effect=AssertionError):
            result = replay.read_parent(package, parent)
        self.assertEqual(result[4].name, 'checkpoint_000150.pt')
        self.assertTrue(result[0].IMPORT_IS_STDLIB_ONLY)
        replay.assert_unchanged(result[5])

    def test_corrupted_checkpoint_rejected(self):
        package, parent, identity, _, _ = self.fixture()
        (parent/'checkpoints/checkpoint_000150.pt').write_bytes(b'changed')
        with patch.object(replay, 'IDENTITY', identity), self.assertRaisesRegex(ValueError, 'sidecar/hash'):
            replay.read_parent(package, parent)

    def test_incomplete_journal_rejected(self):
        package, parent, identity, _, rows = self.fixture()
        rows[155]['optimizer_step'] = 155
        (parent/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch.object(replay, 'IDENTITY', identity), self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
            replay.read_parent(package, parent)

    def test_mismatched_dataset_order_fails_without_prepare(self):
        _, _, _, manifest, rows = self.fixture()
        def batches(runtime, names, seed):
            for row in rows:
                yield 0, (None, [row['image']+'wrong'])
        exp = types.SimpleNamespace(batches=batches)
        with self.assertRaisesRegex(ValueError, 'data order'):
            replay.reconstruct_iterator(exp, object(), manifest, rows, 150)

    def test_reconstructed_iterator_starts_at151(self):
        _, _, _, manifest, rows = self.fixture()
        exp = types.SimpleNamespace(batches=lambda runtime, names, seed:
                    iter((r['epoch'], (None, [r['image']])) for r in rows))
        iterator = replay.reconstruct_iterator(exp, object(), manifest, rows, 150)
        self.assertEqual(next(iterator)[1][-1][0], rows[150]['image'])

    def test_optimizer_restore_preserves_next_update(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.tensor([[.1,.2,.3]])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x).square().sum().backward()
            optimizer.step()
        saved = copy.deepcopy(dict(step=2, model=model.state_dict(), optimizer=optimizer.state_dict(), rng={}))
        other = torch.nn.Linear(3, 2)
        resumed = torch.optim.AdamW(other.parameters(), lr=.5)
        replay.restore_model_optimizer(torch, types.SimpleNamespace(state_fingerprint=fingerprint), other, resumed, saved, 2)
        for head, opt in ((model, optimizer), (other, resumed)):
            opt.zero_grad(set_to_none=True)
            head(x).square().sum().backward()
            opt.step()
        self.assertEqual(fingerprint(model.state_dict()), fingerprint(other.state_dict()))
        self.assertEqual(fingerprint(optimizer.state_dict()), fingerprint(resumed.state_dict()))

    def test_backward_failure_never_updates_optimizer(self):
        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()
            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError('first backward failure')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.4,.6]))
            def forward(self, **kwargs):
                return {'probabilities': Failing.apply(self.weight).gather(0, torch.tensor([0,1]))}
        model = Model()
        optimizer = torch.optim.AdamW(model.parameters())
        example = types.SimpleNamespace(common={'q':torch.ones(1), 'ids':torch.zeros(1,dtype=torch.long)},
                                        target=torch.zeros(1,dtype=torch.long))
        support = {k:torch.ones(1) for k in ('route_q','route_ids','support_ids')}
        factory = types.SimpleNamespace(prepare=lambda *args:{'d':support})
        runtime = types.SimpleNamespace(loss=lambda p,t:p.square().sum())
        state = types.SimpleNamespace(lr_at_step=lambda *args,**kwargs:.001)
        status = replay.initial_status(self.root)
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'first backward failure'):
            replay.observed_update(None,model,optimizer,example,runtime,factory,state,None,
                                   step=151,output=self.root,status=status,multiplier=.2)
        self.assertTrue(torch.equal(before,model.weight))
        self.assertEqual(status['optimizer_updates_returned'],0)
        self.assertEqual(len(optimizer.state),0)
        self.assertTrue((self.root/'graph_failure_000151.json').is_file())

    def test_diagnostic_status_is_never_formal_completion(self):
        status = replay.initial_status(self.root)
        for key in ('complete','training_complete','full_validation_evidence','full_test_evidence','test_accessed',
                    'historical_failure_resolved','exact_historical_replay','automatic_retry'):
            self.assertIs(status[key],False)
        self.assertEqual(list(range(replay.RESTORED_STEP+1,replay.END_STEP+1)),list(range(151,159)))


if __name__ == '__main__':
    unittest.main()
