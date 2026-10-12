"""Real small V2 frontend contracts plus synthetic Train/Val-loop smoke tests."""
import io
import json
import tempfile
import unittest
import copy
from types import SimpleNamespace
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.common import state_digest, objective
from .generator import structure_reconstruction_loss
from .model import SUITES, capture_inputs, make_branches, member_hashes, gradients, hard_route
from .quality import partition_quality


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.manual_seed(781)
        cls.front = SRACSwin(dim=16).eval().requires_grad_(False)
        cls.source = Readout(16).eval().requires_grad_(False)
        cls.original = state_digest(cls.front), state_digest(cls.source)
        cls.reference, cls.inputs = capture_inputs(cls.front, torch.randn(1,3,64,64), torch.randn(1,3,64,64))
        # Real captured features; varied B is a synthetic routing stress test only.
        cls.reference = dict(cls.reference, building=(torch.arange(256).reshape(1,1,16,16)%2).float())
        cls.target = (torch.arange(4096).reshape(1,64,64)//17%4).long()
        cls.target[:,0] = 255

    @classmethod
    def tearDownClass(cls):
        assert cls.original == (state_digest(cls.front), state_digest(cls.source))

    def test_capture_frozen_real_modalities_and_hook_cleanup(self):
        self.assertEqual(len(self.inputs['sar']), 4)
        self.assertEqual(self.inputs['sar_feature'].shape[-2:], self.reference['shape'])
        for x in (*self.inputs['rgb'], *self.inputs['sar'], self.inputs['sar_feature']):
            self.assertFalse(x.requires_grad); self.assertIsNone(x.grad_fn)
        self.assertFalse(self.front.base.encoder_2._forward_hooks)
        self.assertFalse(self.front.base.fusion_layer_2._forward_hooks)

    def test_every_suite_forward_backward_changes_only_permitted_modules(self):
        for suite in SUITES:
            branches = make_branches(self.front, self.source, suite)
            initial = member_hashes(branches)
            for name, branch in branches.items():
                optimizer = torch.optim.AdamW((p for p in branch.parameters() if p.requires_grad), lr=1e-4)
                output = branch(self.reference, self.inputs, (64,64))
                torch.testing.assert_close(output['prob'].sum(1), torch.ones(1,64,64), atol=1e-5, rtol=1e-5)
                loss = objective(output['prob'], self.target)
                if name == 'learn_q_structure':
                    loss = loss+.1*structure_reconstruction_loss(output['q'], self.reference['ids'],
                        self.reference['tokens'].shape[1], self.reference['shape'], self.target)
                self.assertTrue(torch.isfinite(loss)); loss.backward()
                optimizer.step()
            report = gradients(branches, self.front)
            final = member_hashes(branches)
            for name, branch in branches.items():
                for member, module in branch.named_children():
                    active = any(p.requires_grad for p in module.parameters())
                    self.assertEqual(report[name][member]['nonzero'], active, (suite,name,member))
                    self.assertEqual(initial[name][member] != final[name][member], active)

    def test_initial_replay_and_equal_capacity(self):
        a = make_branches(self.front, self.source, 'structure')
        outputs = [branch(self.reference,self.inputs,(64,64)) for branch in a.values()]
        for value in outputs:
            torch.testing.assert_close(value['q'],self.reference['q'],atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(value['prob'],outputs[0]['prob'],atol=2e-5,rtol=2e-4)
        dual = make_branches(self.front,self.source,'multimodal')
        self.assertEqual(sum(p.numel() for p in dual['rgb_rgb'].parameters()),
                         sum(p.numel() for p in dual['rgb_sar'].parameters()))
        rgb = dual['rgb_rgb'](self.reference,self.inputs,(64,64))
        torch.testing.assert_close(rgb['q'],self.reference['q'],atol=2e-5,rtol=2e-4)
        self.assertEqual(state_digest(dual['rgb_sar'].decoder),state_digest(dual['rgb_sar'].second_decoder))
        for x,y in zip(dual['rgb_sar'].decoder.parameters(),dual['rgb_sar'].second_decoder.parameters()):
            self.assertNotEqual(x.untyped_storage().data_ptr(),y.untyped_storage().data_ptr())

    def test_support_keeps_read_q_identical_and_shared_budget(self):
        branches = make_branches(self.front,self.source,'support')
        values = {n:b(self.reference,self.inputs,(64,64)) for n,b in branches.items()}
        for n,r in values.items():
            self.assertTrue(torch.equal(r['q'],self.reference['q']))
            self.assertTrue(torch.equal(r['read_route']['q'],values['hard']['read_route']['q']))
        a,b=(values[n]['exchange_diagnostics']['actual_exchange_mean'] for n in ('uniform','sar'))
        self.assertAlmostEqual(a,b,places=6)

    def test_quality_all_background_does_not_invent_foreground_evidence(self):
        route=hard_route(self.reference,self.reference['q'],.5)
        quality=partition_quality(route,torch.zeros_like(self.target))
        self.assertEqual(quality['foreground_mass'],0)
        self.assertGreater(quality['class_0_mass'],0)
        self.assertLessEqual(quality['hard_fragmented_regions'],quality['hard_used_regions'])

    def test_baseline_reproduction_rejects_offsetting_class_drift(self):
        from .train import verify_baselines
        metric=dict(valid_miou=.5, iou=[.9,.6,.4,.5],
                    precision=[.9,.7,.6,.7], recall=[.9,.7,.6,.7])
        expected={n:copy.deepcopy(metric) for n in ('old_pixel','old_region','hard')}
        actual={n:copy.deepcopy(metric) for n in ('old_pixel','old_region','source_hard')}
        args=SimpleNamespace(smoke=False)
        verify_baselines({'summary':{'final_metrics':expected}}, {'metrics':actual}, args)
        actual['source_hard']['iou'][1]+=.01
        actual['source_hard']['iou'][2]-=.01
        with self.assertRaises(ValueError):
            verify_baselines({'summary':{'final_metrics':expected}}, {'metrics':actual}, args)

    def test_actual_twenty_step_loops_with_mocked_source_and_data(self):
        """Run real model branches/optimizer/checkpoints/evaluation, not server data."""
        from . import train
        for suite in SUITES:
            bundle=dict(front=self.front, heads={'hard':self.source}, device=torch.device('cpu'),
                train=['train_a'],val=['val_a','val_b'], summary={},
                provenance={'source_head_step':5000}, manifest={'arguments':{'epsilon':.05,'threshold':.5}})
            batch=(torch.zeros(1,3,64,64),torch.zeros(1,3,64,64),torch.zeros_like(self.target),self.target,['val_a'])
            def fake_loader(dataset,names,train=False,batch=1,workers=0,seed=0):
                self.assertEqual(batch,1)
                def item(name):
                    return (torch.zeros(1,3,64,64),torch.zeros(1,3,64,64),
                            torch.zeros_like(self.target),self.target,[name])
                return [item(n) for n in names]
            def check_source(_bundle):
                self.assertEqual(self.original,(state_digest(self.front),state_digest(self.source)))
                self.assertTrue(all(p.grad is None for p in self.front.parameters()))
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                out=Path(tmp)/suite
                stack.enter_context(mock.patch.object(train,'load_source',return_value=bundle))
                stack.enter_context(mock.patch.object(train,'loader',side_effect=fake_loader))
                stack.enter_context(mock.patch.object(train,'parse_batch',side_effect=lambda b,d:b))
                stack.enter_context(mock.patch.object(train,'capture_inputs',return_value=(self.reference,self.inputs)))
                stack.enter_context(mock.patch.object(train,'assert_source',side_effect=check_source))
                stack.enter_context(mock.patch.object(train,'verify_original_readout',return_value=0.))
                stack.enter_context(mock.patch.object(train,'implementation_hash',return_value='test-only'))
                with redirect_stdout(io.StringIO()):
                    train.main(['--suite',suite,'--checkpoint','mock','--run_root','mock','--dataset','mock',
                        '--train_list','mock','--val_list','mock','--output',str(out),'--steps','20',
                        '--eval_every','20','--batch','1','--accum','16','--workers','0','--device','cpu','--smoke'])
                summary=json.loads((out/'summary.json').read_text())
                self.assertTrue(summary['complete']); self.assertTrue(summary['smoke'])
                self.assertEqual(summary['final_step'],20)
                self.assertFalse(summary['test_accessed'])
                self.assertTrue((out/'audit_final.json').is_file())
                for field in ('partition_quality_totals','base_partition_quality_totals'):
                    self.assertEqual(set(summary[field]),set(SUITES[suite]))
                    self.assertTrue(all(r['image_count']==2 for r in summary[field].values()))
                for name in SUITES[suite]:
                    payload=torch.load(out/(name+'_last.pth'),map_location='cpu',weights_only=False)
                    self.assertEqual(payload['step'],20)
                    self.assertFalse(any(k.startswith(('front.','base.')) for k in payload['branch']))
                    self.assertNotIn('optimizer',payload)


if __name__=='__main__':
    unittest.main()
