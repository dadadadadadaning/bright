"""CPU-only Q path contracts, including a real small SRACSwin V2 forward."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
from .common import objective, state_digest
from .q_model import (
    capture_q_inputs, decoder_q, gradient_report, make_branches, verify_initial_replay,
)
from .train_q import checkpoint_payload


class QContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(3411)
        cls.front = SRACSwin(dim=16).eval().requires_grad_(False)
        cls.source = Readout(16).eval().requires_grad_(False)
        cls.front_hash = state_digest(cls.front)
        cls.source_hash = state_digest(cls.source)
        cls.front_buffers = {k: v.clone() for k, v in cls.front.named_buffers()}
        cls.reference, cls.pyramid = capture_q_inputs(cls.front,
            torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

    @classmethod
    def tearDownClass(cls):
        if state_digest(cls.front) != cls.front_hash or state_digest(cls.source) != cls.source_hash:
            raise AssertionError('Real source model parameters or buffers changed')

    def setUp(self):
        torch.manual_seed(3412)
        self.branches = make_branches(self.front, self.source)

    def test_real_small_front_initial_q_and_full_probability_replay(self):
        for mode in ('hard', 'exchange'):
            report = verify_initial_replay(self.branches, self.source, self.reference,
                self.pyramid, (64, 64), pool_mode=mode, read_mode=mode)
            self.assertTrue(report['passed'])
            self.assertLessEqual(report['learn_q_q_max_abs'], 2e-5)
            self.assertLessEqual(report['legacy_prob_max_abs'], 2e-5)
        for feature in (*self.pyramid, self.reference['features']['roi'], self.reference['building']):
            self.assertFalse(feature.requires_grad)
            self.assertIsNone(feature.grad_fn)

    def test_independent_identical_complete_decoder_and_readout_initialization(self):
        for member in ('decoder', 'readout'):
            source = self.front.decoder if member == 'decoder' else self.source
            for name, branch in self.branches.items():
                self.assertEqual(state_digest(getattr(branch, member)), state_digest(source))
            for a, b in zip(getattr(self.branches['fixed_q'], member).parameters(),
                            getattr(self.branches['learn_q'], member).parameters()):
                self.assertTrue(torch.equal(a, b))
                self.assertNotEqual(a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        self.assertFalse(any(p.requires_grad for p in self.branches['fixed_q'].decoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.branches['learn_q'].decoder.parameters()))
        self.assertTrue(all(p.requires_grad for b in self.branches.values() for p in b.readout.parameters()))

    def test_existing_loss_reaches_partition_weights_and_changes_them(self):
        before = {name: {member: state_digest(getattr(branch, member))
                        for member in ('decoder', 'readout')} for name, branch in self.branches.items()}
        target = (torch.arange(64*64).reshape(1, 64, 64)//17 % 4).long()
        target[:, 0] = 255
        outputs = {}
        for name, branch in self.branches.items():
            branch.train()
            optimizer = torch.optim.AdamW((p for p in branch.parameters() if p.requires_grad),
                                         lr=1e-4, weight_decay=.005)
            outputs[name] = branch(self.reference, self.pyramid, target.shape[-2:])
            if name == 'learn_q':
                outputs[name]['q'].retain_grad()
            loss = objective(outputs[name]['prob'], target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            if name == 'learn_q':
                q_gradient = outputs[name]['q'].grad
                self.assertIsNotNone(q_gradient)
                self.assertTrue(torch.isfinite(q_gradient).all())
                self.assertGreater(float(q_gradient.abs().sum()), 0.)
            torch.nn.utils.clip_grad_norm_((p for p in branch.parameters() if p.requires_grad),
                                          1., error_if_nonfinite=True)
            optimizer.step()
        report = gradient_report(self.branches, self.front)
        self.assertFalse(report['fixed_q']['decoder']['nonzero'])
        self.assertTrue(report['learn_q']['decoder']['nonzero'])
        self.assertEqual(before['fixed_q']['decoder'], state_digest(self.branches['fixed_q'].decoder))
        self.assertNotEqual(before['learn_q']['decoder'], state_digest(self.branches['learn_q'].decoder))
        for name, branch in self.branches.items():
            self.assertTrue(report[name]['readout']['nonzero'])
            self.assertNotEqual(before[name]['readout'], state_digest(branch.readout))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.front.parameters()))
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))
        for key, value in self.front.named_buffers():
            self.assertTrue(torch.equal(value, self.front_buffers[key]), key)

    def test_feature_pyramid_building_and_fixed_reference_q_are_detached(self):
        pyramid = tuple(x.clone().requires_grad_() for x in self.pyramid)
        roi = self.reference['features']['roi'].clone().requires_grad_()
        building = self.reference['building'].clone().requires_grad_()
        reference_q = self.reference['q'].clone().requires_grad_()
        reference = dict(self.reference, q=reference_q, building=building,
                         features=dict(self.reference['features'], roi=roi))
        for branch in self.branches.values():
            output = branch(reference, pyramid, (32, 32), pool_mode='exchange', read_mode='hard')
            (-output['prob'][:, 2].clamp_min(1e-8).log().mean()).backward()
        self.assertTrue(all(x.grad is None for x in (*pyramid, roi, building, reference_q)))
        self.assertGreater(gradient_report(self.branches, self.front)['learn_q']['decoder']['norm'], 0.)

    def test_clustering_receives_zero_building_then_routing_uses_fixed_building(self):
        from . import q_model
        changed = dict(self.reference, building=1-self.reference['building'])
        with mock.patch.object(q_model, 'iterative_assignment', wraps=q_model.iterative_assignment) as call:
            original = decoder_q(self.branches['learn_q'].decoder, self.pyramid,
                                 self.reference, self.front.config)
            flipped = decoder_q(self.branches['learn_q'].decoder, self.pyramid,
                                changed, self.front.config)
            self.assertEqual(call.call_count, 2)
            for entry in call.call_args_list:
                self.assertEqual(int(torch.count_nonzero(entry.args[3])), 0)
        torch.testing.assert_close(original, flipped, atol=0, rtol=0)
        for value in (0., 1.):
            reference = dict(self.reference, building=torch.full_like(self.reference['building'], value))
            output = self.branches['learn_q'](reference, self.pyramid, (32, 32))
            absent = output['pool_route']['q'][:, :9] if value else output['pool_route']['q'][:, 9:]
            self.assertEqual(int(torch.count_nonzero(absent)), 0)

    def test_replay_rejects_wrong_reference_q(self):
        changed = dict(self.reference, q=self.reference['q'].roll(1, dims=1))
        with self.assertRaises(AssertionError):
            verify_initial_replay(self.branches, self.source, changed, self.pyramid, (64, 64))

    def test_gradient_audit_rejects_disconnected_learned_partition(self):
        for branch in self.branches.values():
            for parameter in branch.readout.parameters():
                parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(RuntimeError, 'disconnected decoder'):
            gradient_report(self.branches, self.front)

    def test_only_decoder_readout_saved_no_memory_front_or_optimizer(self):
        for name, branch in self.branches.items():
            self.assertEqual(set(dict(branch.named_children())), {'decoder', 'readout'})
            self.assertTrue(all(key.startswith(('decoder.', 'readout.')) for key in branch.state_dict()))
            payload = checkpoint_payload(branch, name, 20, {'valid_miou': .1}, {'no_memory': True})
            self.assertEqual(set(payload), {'format', 'name', 'step', 'decoder', 'readout', 'metrics', 'manifest'})
            stream = io.BytesIO()
            torch.save(payload, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=True)
            branch.decoder.load_state_dict(restored['decoder'], strict=True)
            branch.readout.load_state_dict(restored['readout'], strict=True)
            self.assertEqual(restored['format'], 'ROI_Q_FOLLOWUP_V1')

    def test_mocked_cpu_twenty_step_main_checkpoints_diagnostics_and_report(self):
        """Actual paired EBS16 loop; reuse captured real 64x64 frozen inputs.

        Source/data loading is synthetic. Decoder/readout backward, AdamW,
        initialization replay, validation, checkpoint IO and reports are real.
        No benchmark images, user output directories or CUDA are accessed.
        """
        from . import train_q, report
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        bundle = dict(front=self.front, heads={'hard': self.source},
                      train=train_names, val=val_names, device=torch.device('cpu'),
                      summary={}, provenance={},
                      manifest={'arguments': {'epsilon': .05, 'threshold': .5}})
        loader_calls, capture_calls = [], []

        def assert_source(received):
            self.assertIs(received, bundle)
            self.assertEqual(state_digest(self.front), self.front_hash)
            self.assertEqual(state_digest(self.source), self.source_hash)
            for model in (self.front, self.source):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

        def loader(dataset, names, train=False, batch=1, workers=0, seed=0):
            loader_calls.append((train, list(names), batch))
            batches = []
            for start in range(0, len(names), batch):
                current = list(names[start:start+batch])
                pre = torch.zeros(len(current), 3, 64, 64)
                target = torch.stack([(torch.arange(64*64).reshape(64, 64)//17 +
                                      int('event_b' in name)) % 4 for name in current]).long()
                target[:, 0] = 255
                batches.append((pre, pre.clone(), torch.zeros_like(target), target, current))
            return batches

        batched_reference = {}
        for key, value in self.reference.items():
            if isinstance(value, dict):
                batched_reference[key] = {n: x.expand(2, *x.shape[1:]).clone() for n, x in value.items()}
            elif torch.is_tensor(value) and value.ndim and value.shape[0] == 1:
                batched_reference[key] = value.expand(2, *value.shape[1:]).clone()
            else:
                batched_reference[key] = value
        batched_pyramid = tuple(x.expand(2, *x.shape[1:]).clone() for x in self.pyramid)

        def capture(model, pre, post):
            self.assertIs(model, self.front)
            capture_calls.append(len(pre))
            if len(pre) == 1:
                return self.reference, self.pyramid
            self.assertEqual(len(pre), 2)
            return batched_reference, batched_pyramid

        with tempfile.TemporaryDirectory(prefix='roi_q_cpu_test_') as temporary, ExitStack() as stack:
            output = Path(temporary)/'paired'
            for name, replacement in (
                ('load_source', mock.Mock(return_value=bundle)),
                ('assert_source', assert_source),
                ('source_identity', mock.Mock(return_value={'scope': 'synthetic CPU integration'})),
                ('loader', loader), ('capture_q_inputs', capture),
                ('parse_batch', lambda batch, device: batch),
            ):
                stack.enter_context(mock.patch.object(train_q, name, replacement))
            argv = ['--checkpoint', 'synthetic_front', '--run_root', 'synthetic_source',
                    '--dataset', 'synthetic', '--train_list', 'train_set.txt', '--val_list', 'val_set.txt',
                    '--output', str(output), '--device', 'cpu', '--workers', '0', '--smoke',
                    '--steps', '20', '--eval_every', '20', '--batch', '2', '--accum', '8']
            with redirect_stdout(io.StringIO()):
                train_q.main(argv)
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['smoke'])
            self.assertFalse(summary['test_accessed'])
            self.assertEqual(summary['final_step'], 20)
            self.assertTrue(summary['learned_q_gradient_verified'])
            self.assertFalse(summary['weights_changed']['fixed_q']['decoder'])
            self.assertTrue(summary['weights_changed']['learn_q']['decoder'])
            self.assertEqual(summary['nonzero_gradient_steps']['learn_q']['decoder'], 20)
            self.assertGreater(summary['learned_base_q_max_abs_change'], 0.)
            audit = json.loads((output/'audit_final.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['image_count'], 2)
            self.assertIn('fixed_q__learn_q', audit['groups']['all']['comparisons'])
            self.assertEqual(set(audit['groups']['all']['metrics']),
                             {'old_pixel', 'old_region', 'fixed_q', 'learn_q'})
            self.assertIn('roi_false_negative', audit['groups'])
            self.assertIn('Damaged__boundary', audit['groups'])
            for name in ('fixed_q', 'learn_q'):
                for suffix in ('best', 'last'):
                    payload = torch.load(output/(name+'_'+suffix+'.pth'), map_location='cpu', weights_only=True)
                    self.assertEqual(payload['step'], 20)
                    self.assertEqual(set(payload), {'format', 'name', 'step', 'decoder', 'readout', 'metrics', 'manifest'})
            training = [json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()]
            self.assertEqual(training[-1]['crops'], 320)
            validation = [json.loads(line) for line in (output/'validation.jsonl').read_text().splitlines()]
            self.assertEqual(validation[-1]['step'], 20)
            self.assertEqual(validation[-1]['image_count'], 2)
            self.assertIn('routing', validation[-1])
            rendered = report.render(output)
            self.assertIn('PRIMARY learn_q minus fixed_q', rendered)
            self.assertIn('execution evidence only', rendered)
            stream = io.StringIO()
            with mock.patch.object(sys, 'argv', ['report', '--root', str(output)]), redirect_stdout(stream):
                report.main()
            self.assertEqual(stream.getvalue().strip(), rendered)
        self.assertEqual(loader_calls, [(True, train_names, 2), (False, val_names, 1)])
        self.assertEqual(capture_calls.count(2), 20*8)
        self.assertEqual(capture_calls.count(1), 4)


if __name__ == '__main__':
    unittest.main()
