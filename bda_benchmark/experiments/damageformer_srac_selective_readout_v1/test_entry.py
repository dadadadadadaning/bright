"""CPU integration contracts; synthetic tests are not dataset evidence."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SRACTransformerFront, StructureTokenAdapter, SuperpixelEmbedding,
    SwinLocalSuperpixelAttention, project_region_probabilities, soft_assignment_boundary,
)
from . import train
from .models import ARMS, gradient_norms, make_arms
from .runtime import (
    FrozenEvidence, assert_independent, module_fingerprint, objective,
    reproduction_gate, upsample_probability,
)


class TinyA1(nn.Module):
    """Same 256-to-4 head and lower-resolution feature interface as real A1."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(6, 256, 1), nn.BatchNorm2d(256), nn.GELU())
        self.clf_1 = nn.Conv2d(256, 2, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)
        self.forward_calls = 0

    def forward(self, pre, post, return_aux=False):
        self.forward_calls += 1
        feature = self.feature(F.avg_pool2d(torch.cat((pre, post), 1), 2))
        loc = self.clf_1(feature)
        building = loc.softmax(1)[:, 1:2]
        fused = feature * (0.1 + 0.9 * building)
        damage = self.clf_2(fused)
        size = pre.shape[-2:]
        output = (F.interpolate(loc, size, mode='bilinear', align_corners=False),
                  F.interpolate(damage, size, mode='bilinear', align_corners=False))
        if return_aux:
            return (*output, dict(fused_roi=fused, p_building=building))
        return output


class TinyFront(nn.Module):
    """Real parent Q/embedding primitives with inexpensive image encoders."""

    def __init__(self):
        super().__init__()
        self.config = dict(dim=16)
        self.base = TinyA1()
        self.decoder = nn.Conv2d(256, 8, 1)
        self.partition = SwinLocalSuperpixelAttention(8, 8, stride=2, iterations=1)
        self.adapter = StructureTokenAdapter(256, 16)
        self.embedding = SuperpixelEmbedding(256, 16, 16)
        self.region_classifier = nn.Linear(16, 4)

    def forward(self, pre, post):
        loc, old, aux = self.base(pre, post, return_aux=True)
        raw, building = aux['fused_roi'], aux['p_building']
        partition = self.partition(self.decoder(raw), building)
        boundary = soft_assignment_boundary(partition['q'], partition['ids'], partition['shape'])
        adapted, content = self.adapter(raw, building, boundary, partition['entropy_map'])
        tokens, mass, geometry, valid = self.embedding(
            adapted, raw, partition['q'], partition['ids'], partition['shape'], partition['count'],
        )
        logits = self.region_classifier(tokens)
        probability = project_region_probabilities(
            logits, partition['q'], partition['ids'], partition['shape'], pre.shape[-2:],
        )
        return dict(**partition, old_pixel=old, loc=loc, prob=probability, tokens=tokens,
                    mass=mass, geometry=geometry, valid_regions=valid, region_logits=logits,
                    soft_boundary=boundary, adapted_content=adapted, content_only=content)


def batch(size=2):
    pre, post = torch.randn(size, 3, 8, 8), torch.randn(size, 3, 8, 8)
    labels = (torch.arange(64).reshape(1, 8, 8) // 4 % 4).expand(size, -1, -1).clone()
    return pre, post, (labels > 0).long(), labels, [f'synthetic-event_{i}' for i in range(size)]


class FrozenReadoutIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(823)

    def check_capture_and_initialization(self, front, pre, post):
        source = FrozenEvidence(front)
        fingerprint = module_fingerprint(front)
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            fixed = source(pre, post)
        for key in ('pixel_feature', 'tokens', 'q', 'ids', 'building', 'valid_regions'):
            self.assertFalse(torch.is_inference(fixed[key]), key)
            self.assertFalse(fixed[key].requires_grad, key)
        self.assertEqual(fixed['pixel_feature'].dtype, torch.float32)
        self.assertEqual(fixed['pixel_feature'].shape[1], 256)
        self.assertEqual(fixed['q'].shape[1], 9)
        arms = make_arms(front, score_dim=4, chunk_size=7)
        assert_independent(front, arms)
        self.assertEqual(len({module_fingerprint(arm) for arm in arms.values()}), 1)
        outputs = {name: arm(fixed) for name, arm in arms.items()}
        expected = fixed['old_pixel'].float().softmax(1)
        for name, output in outputs.items():
            torch.testing.assert_close(upsample_probability(output['logits'], pre.shape[-2:]),
                                       expected, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(outputs['q_read']['context'], outputs['content_read']['context'],
                                   rtol=0, atol=0)
        self.assertEqual(outputs['content_read']['routing']['max_route_delta'], 0)
        self.assertEqual(outputs['content_read']['routing']['invalid_route_mass'], 0)
        # A trainable consumer may save captured tensors for backward even after
        # capture was called under both inference mode and CPU autocast.
        outputs['content_read']['logits'].square().mean().backward()
        self.assertIsNotNone(arms['content_read'].classifier.pixel.weight.grad)
        self.assertTrue(all(p.grad is None for p in front.parameters()))
        self.assertEqual(fingerprint, module_fingerprint(front))
        self.assertFalse(any(module.training for module in front.modules()))

    def test_real_small_front_capture_and_a1_identity(self):
        front = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1)
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        self.check_capture_and_initialization(front, pre, post)

    def test_outer_inference_capture_uses_one_source_forward(self):
        front = TinyFront()
        pre, post, *_ = batch()
        self.check_capture_and_initialization(front, pre, post)
        self.assertEqual(front.base.forward_calls, 1)

    def test_three_updates_open_scorer_without_sharing_or_changing_source(self):
        front = TinyFront()
        source = FrozenEvidence(front)
        arms = make_arms(front, score_dim=4, chunk_size=7)
        assert_independent(front, arms)
        initial_source = module_fingerprint(front)
        initial_state = {name: copy.deepcopy(arm.state_dict()) for name, arm in arms.items()}
        optimizers = {name: torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],
                                            lr=0.01, weight_decay=0)
                      for name, arm in arms.items()}
        pre, post, _, target, _ = batch()
        fixed = source(pre, post)
        q_before = fixed['q'].clone()
        readout_norms = []
        reader_norms = []
        for _step in range(3):
            for name, arm in arms.items():
                arm.train()
                optimizers[name].zero_grad(set_to_none=True)
            saved = {}
            for index, (name, arm) in enumerate(arms.items()):
                probability = upsample_probability(arm(fixed)['logits'], target.shape[-2:])
                objective(probability, target).backward()
                for other in ARMS[index + 1:]:
                    self.assertTrue(all(p.grad is None for p in arms[other].parameters()))
                for previous, gradients in saved.items():
                    parameters = dict(arms[previous].named_parameters())
                    for key, value in gradients.items():
                        torch.testing.assert_close(parameters[key].grad, value, rtol=0, atol=0)
                saved[name] = {key: p.grad.clone() for key, p in arm.named_parameters() if p.grad is not None}
            norms = gradient_norms(arms['content_read'])
            readout_norms.append(norms['readout'])
            reader_norms.append(norms['reader'])
            for name, optimizer in optimizers.items():
                other_before = {other: module_fingerprint(arm) for other, arm in arms.items() if other != name}
                optimizer.step()
                for other, fingerprint in other_before.items():
                    self.assertEqual(module_fingerprint(arms[other]), fingerprint)
            self.assertTrue(all(p.grad is None for p in front.parameters()))
            self.assertEqual(initial_source, module_fingerprint(front))
            torch.testing.assert_close(fixed['q'], q_before, rtol=0, atol=0)
        self.assertEqual(readout_norms[0], 0)
        self.assertGreater(max(readout_norms[1:]), 0)
        self.assertGreater(max(reader_norms[1:]), 0)
        for name, arm in arms.items():
            self.assertFalse(torch.equal(arm.classifier.pixel.weight,
                                         initial_state[name]['classifier.pixel.weight']))
        for name in ('p', 'q_read'):
            torch.testing.assert_close(arms[name].readout.phi.weight,
                                       initial_state[name]['readout.phi.weight'], rtol=0, atol=0)
        self.assertGreater(float(arms['content_read'].readout.phi.weight.detach().abs().sum()), 0)
        self.assertGreater(float(arms['content_read'].readout.psi.weight.grad.abs().sum()), 0)

    def test_ignore_only_objective_keeps_a_zero_gradient_graph(self):
        logits = torch.randn(2, 4, 3, 3, requires_grad=True)
        loss = objective(logits.softmax(1), torch.full((2, 3, 3), 255, dtype=torch.long))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


class TrainingEntryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(179)

    @staticmethod
    def arguments(output):
        return Namespace(checkpoint='synthetic-snapshot.pth', a1_checkpoint='synthetic-a1.pth',
                         dataset='never-opened', train_list='train_set.txt', val_list='val_set.txt',
                         output=str(output), steps=2, eval_every=2, batch=2, accum=8, workers=0,
                         seed=42, score_dim=4, chunk_size=7, lr=0.001, weight_decay=0.0,
                         amp='none', device='cpu', smoke=False)

    @staticmethod
    def mock_inputs(stack, front):
        payload = dict(valid_miou=0.5, load_report=dict(expected_valid_miou=0.5))
        provenance = dict(diagnostic_only=True, reusable_as_parent=False, test_accessed=False,
                          smoke=False, synthetic_test_only=True)
        stack.enter_context(mock.patch.object(train, 'lists', return_value=(['synthetic-train'], ['synthetic-val'])))
        stack.enter_context(mock.patch.object(train, 'load_fixed', return_value=(front, payload, provenance)))
        training, validation = [batch(2)], [batch(1)]
        def fake_loader(_dataset, _names, train=False, **_kwargs):
            return training if train else validation
        return stack.enter_context(mock.patch.object(train, 'loader', side_effect=fake_loader))

    def test_reproduction_rejection_saves_step0_without_optimizer_update(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            out = Path(temp) / 'rejected'
            front = TinyFront()
            before = module_fingerprint(front)
            self.mock_inputs(stack, front)
            stack.enter_context(mock.patch.object(train, 'reproduction_gate',
                                                  return_value=dict(passed=False, reason='synthetic mismatch')))
            step = stack.enter_context(mock.patch.object(torch.optim.AdamW, 'step', autospec=True))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'Step-0'):
                train.run(self.arguments(out))
            step.assert_not_called()
            evidence = json.loads((out / 'step0.json').read_text())
            self.assertFalse(evidence['reproduction']['passed'])
            self.assertIn('initial_max_abs_probability_error_to_a1', evidence)
            self.assertFalse((out / 'training.jsonl').exists())
            self.assertFalse(json.loads((out / 'manifest.json').read_text())['complete'])
            self.assertEqual(module_fingerprint(front), before)

    def test_two_step_entry_writes_diagnostic_checkpoints_with_32_crops(self):
        # Formal reproduction is deliberately mocked: this verifies entry-point
        # mechanics only and cannot be cited as native A1 or Val450 evidence.
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            out = Path(temp) / 'mechanics'
            front = TinyFront()
            before = module_fingerprint(front)
            load = self.mock_inputs(stack, front)
            stack.enter_context(mock.patch.object(train, 'reproduction_gate',
                                                  return_value=dict(passed=True, synthetic_test_only=True)))
            with contextlib.redirect_stdout(io.StringIO()):
                train.run(self.arguments(out))
            summary = json.loads((out / 'summary.json').read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['source_weights_and_buffers_unchanged'])
            self.assertTrue(summary['independent_gradients_verified'])
            self.assertTrue(summary['diagnostic_only'])
            self.assertFalse(summary['reusable_as_parent'])
            self.assertFalse(summary['test_accessed'])
            self.assertTrue(manifest['synthetic_test_only'])
            self.assertEqual(summary['crops_per_arm'], 32)
            self.assertEqual(summary['final_step'], 2)
            self.assertEqual(manifest['effective_batch'], 16)
            rows = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
            self.assertEqual([row['crops_per_arm'] for row in rows], [16, 32])
            self.assertEqual(front.base.forward_calls, 18)  # 16 microbatches + two Val images.
            self.assertEqual(module_fingerprint(front), before)
            self.assertEqual(load.call_count, 2)
            for name in ARMS:
                for prefix in ('best', 'last'):
                    state = torch.load(out / f'{prefix}_{name}.pth', map_location='cpu', weights_only=False)
                    self.assertTrue(state['run_complete'])
                    self.assertEqual(state['completed_steps'], 2)
                    self.assertFalse(state['reusable_as_parent'])
                    self.assertTrue(state['manifest']['diagnostic_only'])
                    self.assertNotIn('optimizer', state)
                    self.assertFalse(any(key.startswith('base.') for key in state['model']))

    def test_smoke_budget_and_full_val_reproduction_are_separate(self):
        args = self.arguments(Path('this-path-is-not-created-by-this-test'))
        args.smoke = True
        train.validate_args(args)
        self.assertEqual((args.steps, args.eval_every), (20, 20))
        payload = dict(valid_miou=0.6, load_report=dict(expected_valid_miou=0.61))
        result = dict(image_count=450, metrics=dict(a1=dict(valid_miou=0.61), region=dict(valid_miou=0.6)))
        self.assertTrue(reproduction_gate(result, payload, False)['passed'])
        result['image_count'] = 2
        self.assertFalse(reproduction_gate(result, payload, False)['passed'])
        self.assertIsNone(reproduction_gate(result, payload, True)['passed'])


if __name__ == '__main__':
    unittest.main()
