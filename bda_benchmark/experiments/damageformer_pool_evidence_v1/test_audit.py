"""CPU synthetic integration contracts; no real checkpoints or dataset required."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import candidate_map
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import route_from_q
from bda_benchmark.experiments.damageformer_partition_learning_v1 import reproduction
from . import audit
from .aggregate import PoolAccumulator


def _frozen_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _fixture():
    generator = torch.Generator().manual_seed(916)
    height, width, stride = 4, 4, 2
    ids, valid, gh, gw = candidate_map(height, width, stride, torch.device('cpu'))
    q = torch.rand((1, 9, height*width), generator=generator)*valid[None]
    q = q/q.sum(1, keepdim=True)
    building = torch.linspace(.02, .98, height*width).reshape(1, 1, height, width)
    feature = torch.randn((1, 256, height, width), generator=generator)
    head = Readout(dim=8).requires_grad_(False).eval()
    front = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3)).requires_grad_(False).eval()
    gt = (torch.arange(64).reshape(1, 8, 8)//4) % 4
    gt[0, 0, 0] = 255
    baseline = gt.clamp(0, 3).clone()
    baseline[:, :, ::3] = (baseline[:, :, ::3] + 1) % 4
    pixel_logits = torch.nn.functional.one_hot(baseline, 4).permute(0, 3, 1, 2).float()*4
    reference = dict(q=q, ids=ids, shape=(height, width),
                     tokens=torch.zeros(1, gh*gw, 8), building=building,
                     features={'roi': feature}, pixel=pixel_logits,
                     prob=pixel_logits.softmax(1))
    bundle = dict(front=front, heads={'hard': head}, device=torch.device('cpu'),
                  manifest={'arguments': {'threshold': .5, 'epsilon': .05}})
    pre = torch.zeros(1, 3, 8, 8)
    batch = (pre, pre.clone(), None, gt, ['bata-explosion_sample'])
    return bundle, reference, batch


class FrozenEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_readout_evaluation_preserves_parameters_buffers_and_q(self):
        bundle, reference, batch = _fixture()
        states = {name: _frozen_state(model) for name, model in
                  {'front': bundle['front'], 'head': bundle['heads']['hard']}.items()}
        original_q = reference['q'].clone()
        original_building = reference['building'].clone()
        def capture_front(front, pre, post):
            # Exercise real frozen Conv/BatchNorm without constructing Swin.
            front(pre)
            return reference
        with patch.object(audit.source, 'capture', side_effect=capture_front), \
                patch.object(audit.source, 'parse_batch', side_effect=lambda value, device: value):
            result = audit.evaluate(bundle, [batch])
        self.assertEqual(result['image_count'], 1)
        self.assertFalse(result['test_accessed'])
        self.assertLess(result['replay_probability_max_abs'], 2e-5)
        self.assertIn('pooling_evidence', result)
        self.assertEqual(result['groups']['all']['metrics']['source_hard']['count'], 63)
        for name, model in {'front': bundle['front'], 'head': bundle['heads']['hard']}.items():
            self.assertFalse(model.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, states[name][key], atol=0, rtol=0)
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in model.parameters()))
        torch.testing.assert_close(reference['q'], original_q, atol=0, rtol=0)
        torch.testing.assert_close(reference['building'], original_building, atol=0, rtol=0)

    def test_disjoint_complete_slot_bins_conserve_full_image_harm_rescue(self):
        bundle, reference, batch = _fixture()
        gt = batch[3]
        route = route_from_q(reference['q'], reference['ids'], reference['tokens'].shape[1],
                             reference['shape'], reference['building'])
        head, feature = bundle['heads']['hard'], reference['features']['roi']
        with torch.no_grad():
            attention = head.embedding.score(feature).sigmoid()
            evidence = audit.region_evidence(route, attention, gt)
            candidate = head(feature, route, gt.shape[-2:]).argmax(1)
        baseline = reference['pixel'].argmax(1)
        for stats in evidence.values():
            even = (torch.arange(route['k'])[None] % 2 == 0).expand_as(stats['mass'])
            # Deliberately cover every slot, including empty ones (zero read weight).
            stats['slot_masks'] = {'test_even': even, 'test_odd': ~even}
        accumulator = PoolAccumulator(.5)
        accumulator.update(evidence, route, gt, baseline, candidate, reference['building'])
        rows = accumulator.finalize()['associations']
        valid = (gt >= 0) & (gt < 4)
        expected = dict(weighted_pixels=int(valid.sum()),
                        harmed=int((valid & (baseline == gt) & (candidate != gt)).sum()),
                        rescued=int((valid & (baseline != gt) & (candidate == gt)).sum()))
        expected['net'] = expected['rescued']-expected['harmed']
        for path in ('stats', 'attention'):
            for key, total in expected.items():
                observed = sum(rows[f'{path}/test_{name}/all'][key] for name in ('even', 'odd'))
                self.assertAlmostEqual(observed, total, delta=2e-4,
                                       msg=f'{path}/{key}: disjoint complete bins must conserve evidence')


def _metrics():
    return dict(valid_miou=.5, iou=[.5]*4, precision=[.5]*4, recall=[.5]*4,
                count=16, support=[4]*4)


def _fake_main_bundle():
    expected = {name: _metrics() for name in ('old_pixel', 'old_region', 'hard')}
    return dict(device=torch.device('cpu'), val=list(range(450)), provenance={},
                summary={'final_step': 5000, 'final_metrics': expected})


def _fake_result(count):
    return dict(image_count=count, groups={'all': {'metrics': {
        name: _metrics() for name in ('old_pixel', 'old_region', 'source_hard')}}},
                test_accessed=False)


class AuditOutputTests(unittest.TestCase):
    def _argv(self, output, smoke):
        args = ['audit', '--checkpoint', 'immutable_front.pth', '--run_root', 'source_run',
                '--dataset', 'synthetic_dataset', '--train_list', 'train_set.txt',
                '--val_list', 'val_set.txt', '--output', str(output), '--device', 'cpu',
                '--workers', '0']
        return args + (['--smoke'] if smoke else [])

    def test_smoke_success_is_not_reported_as_full_val_reproduction(self):
        bundle = _fake_main_bundle()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'smoke'
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', self._argv(out, True)))
                stack.enter_context(patch.object(audit.source, 'load_source', return_value=bundle))
                stack.enter_context(patch.object(audit.source, 'loader', return_value=[]))
                stack.enter_context(patch.object(audit.source, 'assert_source'))
                stack.enter_context(patch.object(audit, 'evaluate', return_value=_fake_result(2)))
                # Keep the actual smoke policy, gate, OutputLock and serialization.
                audit.main()
            summary = json.loads((out/'summary.json').read_text())
            reproduction_report = json.loads((out/'reproduction.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['smoke'])
            self.assertIsNone(summary['reproduction_pass'])
            self.assertEqual(reproduction_report['status'], 'smoke_not_full_val')
            self.assertFalse(summary['weights_updated'])
            self.assertFalse(summary['buffers_updated'])
            self.assertFalse(summary['test_accessed'])
            self.assertTrue((out/'audit_final.json').is_file())
            self.assertFalse(out.with_name(out.name+'.lock').exists())

    def test_failed_formal_reproduction_preserves_candidate_and_incomplete_summary(self):
        bundle, result = _fake_main_bundle(), _fake_result(450)
        policy = reproduction.build_policy(bundle['summary']['final_metrics'], [], [])
        result['groups']['all']['metrics']['source_hard']['valid_miou'] = .8
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'failed'
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', self._argv(out, False)))
                stack.enter_context(patch.object(audit.source, 'load_source', return_value=bundle))
                stack.enter_context(patch.object(audit.source, 'loader', return_value=[]))
                stack.enter_context(patch.object(audit.source, 'assert_source'))
                stack.enter_context(patch.object(audit, 'baseline_policy', return_value=policy))
                stack.enter_context(patch.object(audit, 'evaluate', return_value=result))
                with self.assertRaisesRegex(ValueError, 'Baseline reproduction failed'):
                    audit.main()
            summary = json.loads((out/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['test_accessed'])
            self.assertEqual(summary['error_type'], 'ValueError')
            self.assertTrue((out/'audit_candidate.json').is_file())
            self.assertTrue((out/'initial.json').is_file())
            self.assertFalse(json.loads((out/'reproduction.json').read_text())['passed'])
            self.assertFalse((out/'audit_final.json').exists())
            self.assertFalse(out.with_name(out.name+'.lock').exists())

    def test_existing_output_is_refused_before_loading_source(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'existing'
            out.mkdir()
            sentinel = out/'summary.json'
            # A synthetic test artifact; never touches a real experiment.
            original = b'{"complete": true, "keep": "original"}'
            sentinel.write_bytes(original)
            with patch('sys.argv', self._argv(out, True)), \
                    patch.object(audit.source, 'load_source') as load:
                with self.assertRaises(FileExistsError):
                    audit.main()
            load.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), original)
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
