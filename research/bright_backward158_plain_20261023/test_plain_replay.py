"""CPU contracts for the bounded uninstrumented comparison, not GPU evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import replay


class PlainReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def experiment(self):
        package = Path(os.environ.get('BRIGHT_SEVERITY_PACKAGE',
                                      '/workspace/bright_severity_memory_20261023'))
        path = package / 'experiment.py'
        if not path.is_file():
            self.skipTest('Original update dependency unavailable: '+str(path))
        spec = importlib.util.spec_from_file_location('_plain158_test_original', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def pair_fixture(self):
        package = self.root / 'package'
        package.mkdir()
        (package / 'experiment.py').write_text('STDLIB_IMPORT_ONLY = True\n')
        identity = {'experiment.py': replay.sha(package/'experiment.py')}
        parent = self.root / 'train_original'
        (parent/'checkpoints').mkdir(parents=True)
        plan = dict(format=replay.PARENT_FORMAT, variant='original', effective_mode='original', seed=3407,
                    endpoint_step=2500, read_weight=0, extension_files=identity,
                    scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                    warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
                    post500_lr_multiplier=.2, test_accessed=False)
        train = ['train_'+str(i) for i in range(2918)]
        train[157] = replay.TARGET_IMAGE
        manifest = dict(plan=plan, seed=3407, train_names=train, actual_training_names=train,
                        val_names=['val_'+str(i) for i in range(450)],
                        protocol_info={'source': 'verified'}, runtime_config={'output':str(parent),'seed':3407})
        checkpoint = parent/'checkpoints/checkpoint_000150.pt'
        checkpoint.write_bytes(b'Pair preflight does not deserialize this checkpoint')
        sidecar = dict(step=150, sha256=replay.sha(checkpoint), roundtrip_verified=True, variant='original')
        replay.write(checkpoint.with_suffix('.pt.json'), sidecar)
        replay.write(parent/'plan.json', plan)
        replay.write(parent/'manifest.json', manifest)
        replay.write(parent/'summary.json', dict(phase='failed', last_complete_step=157,
                     failure_phase='update_158', current_image=replay.TARGET_IMAGE, last_checkpoint=sidecar))
        replay.write(parent/'first_failure.json', dict(phase='update_158', image=replay.TARGET_IMAGE,
                                                     last_complete_step=157))
        parent_rows = [dict(step=i, optimizer_step=i, epoch=0, epoch_seed=3407, variant='original',
                           image=train[i-1], original_loss=.1) for i in range(1,158)]
        (parent/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in parent_rows))
        with patch.object(replay, 'IDENTITY', identity):
            parent_data = replay.read_parent(package, parent)
        baseline = self.root/'observed'
        baseline.mkdir()
        (baseline/'runner.pyz').write_bytes(b'pinned observed diagnostic runner')
        runner_sha = replay.sha(baseline/'runner.pyz')
        summary = dict(format=replay.BASELINE_FORMAT, phase='diagnostic_window_complete',
                       diagnostic_complete=True, complete=False, training_complete=False,
                       full_validation_evidence=False, full_test_evidence=False,
                       test_accessed=False, restored_step=150, last_complete_step=158,
                       bounded_endpoint=158, new_complete_updates=8, optimizer_updates_returned=8,
                       original_failed_runs_unchanged=True, historical_failure_resolved=False,
                       exact_historical_replay=False, parent=str(parent), error=None)
        replay.write(baseline/'summary.json', summary)
        replay.write(baseline/'manifest.json', dict(format=replay.BASELINE_FORMAT, parent_manifest=manifest,
                     protocol_info=manifest['protocol_info'], runtime_config={'output':str(baseline),'seed':3407},
                     rng_restored=True, optimizer_state_restored=True, data_order_prefix_verified=True))
        replay.write(baseline/'lineage.json', dict(parent=str(parent), checkpoint=str(checkpoint),
                     checkpoint_sha256=replay.sha(checkpoint), parent_fingerprints=parent_data[-1],
                     steps=list(range(151,159)), target_image=replay.TARGET_IMAGE,
                     formal_checkpoint_written=False, test_accessed=False,
                     historical_failure_resolved=False, exact_historical_replay=False))
        rows = []
        for step in range(151,159):
            compact = dict(complete=True, observed_index_nodes=1, prehook_calls=1,
                           snapshot_bytes=32, coverage_complete=True)
            graph = dict(format='BRIGHT_BACKWARD_SAVED_INDEX_OBSERVER_V1', step=step,
                         phase='observer_step_complete', unexecuted_index_nodes=[],
                         nodes=[dict(prehook_calls=1, statistics={'in_bounds':True})], **compact)
            replay.write(baseline/f'graph_summary_{step:06d}.json', graph)
            rows.append(dict(step=step, optimizer_step=step, epoch=0, image=train[step-1], variant='original',
                             original_loss=.1, training_loss=.1, lr=step*2e-7,
                             grad_norm_preclip=.4, observation_summary=compact))
        (baseline/'diagnostic_updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return package, parent, baseline, identity, runner_sha, rows

    def check_fixture(self, fixture):
        package, parent, baseline, identity, runner_sha, _ = fixture
        with patch.object(replay, 'IDENTITY', identity), \
             patch.object(replay, 'BASELINE_RUNNER_SHA256', runner_sha), \
             patch.object(torch, 'load', side_effect=AssertionError('preflight must not load Torch checkpoint')):
            return replay.check_pair(package, parent, baseline)

    def test_pair_preflight_requires_complete_matching_evidence_without_torch_load(self):
        fixture = self.pair_fixture()
        result = self.check_fixture(fixture)
        self.assertTrue(result[0].STDLIB_IMPORT_ONLY)
        self.assertEqual(len(result[-1]['rows']), 8)
        replay.assert_unchanged(result[-1]['fingerprints'])

    def test_pair_rejects_wrong_runner_sha(self):
        fixture = self.pair_fixture()
        (fixture[2]/'runner.pyz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'runner SHA'):
            self.check_fixture(fixture)

    def test_pair_rejects_partial_baseline(self):
        fixture = self.pair_fixture()
        p = fixture[2]/'summary.json'
        data = replay.read(p)
        data['optimizer_updates_returned'] = 7
        replay.write(p, data)
        with self.assertRaisesRegex(ValueError, 'baseline summary'):
            self.check_fixture(fixture)

    def test_pair_rejects_unrestored_rng_or_bad_parent_fingerprint(self):
        fixture = self.pair_fixture()
        p = fixture[2]/'manifest.json'
        data = replay.read(p)
        data['rng_restored'] = False
        replay.write(p, data)
        with self.assertRaisesRegex(ValueError, 'restore evidence'):
            self.check_fixture(fixture)
        data['rng_restored'] = True
        replay.write(p, data)
        p = fixture[2]/'lineage.json'
        data = replay.read(p)
        data['parent_fingerprints'] = {}
        replay.write(p, data)
        with self.assertRaisesRegex(ValueError, 'fingerprints'):
            self.check_fixture(fixture)

    def test_pair_rejects_unexecuted_graph_or_missing_row(self):
        fixture = self.pair_fixture()
        p = fixture[2]/'graph_summary_000158.json'
        data = replay.read(p)
        data['nodes'][0]['prehook_calls'] = 0
        replay.write(p, data)
        with self.assertRaisesRegex(ValueError, 'index-node coverage'):
            self.check_fixture(fixture)
        data['nodes'][0]['prehook_calls'] = 1
        replay.write(p, data)
        p = fixture[2]/'diagnostic_updates.jsonl'
        p.write_text(''.join(json.dumps(row)+'\n' for row in fixture[-1][:-1]))
        with self.assertRaisesRegex(ValueError, 'eight completed'):
            self.check_fixture(fixture)

    def test_plain_update_calls_original_directly_without_hooks_or_anomaly(self):
        exp = self.experiment()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.2,.7,.4]))
            def forward(self, **kwargs):
                p = self.weight.gather(0, torch.tensor([2,0,1])).sort().values.softmax(0)
                return {'probabilities':p}
        baseline = Model()
        plain = copy.deepcopy(baseline)
        opt_a = torch.optim.AdamW(baseline.parameters(), lr=1e-4, weight_decay=.005)
        opt_b = torch.optim.AdamW(plain.parameters(), lr=1e-4, weight_decay=.005)
        example = types.SimpleNamespace(common={}, target=torch.tensor([0,2]))
        factory = types.SimpleNamespace(prepare=lambda *args:{'d':{}})
        runtime = types.SimpleNamespace(loss=lambda p,t:-p.gather(0,t).log().mean())
        state = types.SimpleNamespace(lr_at_step=lambda step,**kwargs:step*2e-7)
        status = replay.initial_status(self.root)
        with patch.object(torch.autograd, 'detect_anomaly', side_effect=AssertionError('anomaly used')), \
             patch.object(torch.autograd.graph, 'saved_tensors_hooks', side_effect=AssertionError('saved hook used')), \
             patch.object(torch.Tensor, 'register_hook', side_effect=AssertionError('tensor hook used')), \
             patch.object(exp, 'update', wraps=exp.update) as update_call:
            for step in (1,2):
                expected = exp.update(baseline,opt_a,example,runtime,factory,state,None,
                                      step=step,read_weight=0,multiplier=.2)
                actual = replay.plain_update(exp,plain,opt_b,example,runtime,factory,state,None,
                                             step=step,output=self.root,status=status,multiplier=.2)
                self.assertEqual(expected,actual)
                self.assertTrue(torch.equal(baseline.weight,plain.weight))
                for left,right in zip(opt_a.state.values(),opt_b.state.values()):
                    for key in left:
                        self.assertTrue(torch.equal(left[key],right[key]))
            self.assertEqual(update_call.call_count,4)
        self.assertEqual(status['completed_update_calls'],2)
        self.assertEqual(status['optimizer_updates_returned'],2)
        self.assertFalse(list(self.root.glob('graph_*')))

    def test_original_backward_failure_does_not_update_weights(self):
        exp = self.experiment()
        class Failure(torch.autograd.Function):
            @staticmethod
            def forward(ctx,value): return value.clone()
            @staticmethod
            def backward(ctx,gradient): raise RuntimeError('first backward error')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor([.2,.3]))
            def forward(self,**kwargs): return {'probabilities':Failure.apply(self.weight)}
        model=Model()
        optimizer=torch.optim.AdamW(model.parameters())
        before=model.weight.detach().clone()
        example=types.SimpleNamespace(common={},target=None)
        factory=types.SimpleNamespace(prepare=lambda *args:{'d':{}})
        runtime=types.SimpleNamespace(loss=lambda p,t:p.square().sum())
        state=types.SimpleNamespace(lr_at_step=lambda *args,**kwargs:1e-4)
        status=replay.initial_status(self.root)
        with self.assertRaisesRegex(RuntimeError,'first backward error'), \
             exp.capture_training_failure(self.root,status):
            replay.plain_update(exp,model,optimizer,example,runtime,factory,state,None,
                                step=151,output=self.root,status=status,multiplier=.2)
        self.assertTrue(torch.equal(before,model.weight))
        self.assertFalse(optimizer.state)
        self.assertEqual(status['completed_update_calls'],0)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])
        self.assertEqual(replay.read(self.root/'first_failure.json')['error'],
                         'RuntimeError: first backward error')

    def test_failure_after_optimizer_is_not_reported_as_zero_actual_updates(self):
        def failed_update(*args,**kwargs):
            raise RuntimeError('post optimizer finite-parameter check failed')
        status=replay.initial_status(self.root)
        with self.assertRaises(RuntimeError):
            replay.plain_update(types.SimpleNamespace(update=failed_update),None,None,None,None,None,None,None,
                                step=151,output=self.root,status=status,multiplier=.2)
        self.assertEqual(status['completed_update_calls'],0)
        self.assertTrue(status['failed_step_optimizer_execution_unknown'])

    def test_comparison_is_descriptive_without_success_threshold_or_parameter_claim(self):
        rows=self.pair_fixture()[-1]
        changed=copy.deepcopy(rows)
        changed[3]['original_loss']=123.0
        changed[3]['training_loss']=123.0
        changed[3]['grad_norm_preclip']=42.0
        report=replay.compare_rows(rows,changed)
        self.assertTrue(report['comparison_complete'])
        self.assertEqual(report['paired_updates'],8)
        self.assertIsNone(report['numeric_success_threshold'])
        self.assertFalse(report['all_parameters_identical_verified'])
        self.assertFalse(report['full_gradients_compared'])
        self.assertFalse(report['historical_failure_resolved'])
        self.assertGreater(report['pairs'][3]['original_loss']['absolute_difference'],100)
        self.assertFalse(replay.compare_rows(rows,changed[:2])['comparison_complete'])

    def test_status_never_claims_formal_completion_or_no_timing_change(self):
        status=replay.initial_status(self.root)
        for key in ('complete','training_complete','full_test_evidence','full_validation_evidence',
                    'test_accessed','backward_index_observer','anomaly_detection','saved_tensors_hooks_used',
                    'historical_failure_resolved','exact_historical_replay','automatic_retry'):
            self.assertIs(status[key],False,key)
        self.assertTrue(status['observation_changes_timing'])
        self.assertEqual(status['bounded_endpoint'],158)


if __name__ == '__main__':
    unittest.main(verbosity=2)
