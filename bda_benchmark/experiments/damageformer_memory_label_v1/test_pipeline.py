"""Real tiny CPU label reading/training, mocked external data/front/provenance."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from . import experiment as exp
from .report import assess, render
from bda_benchmark.experiments.damageformer_region_memory_v2 import build_bank
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bank_audit_and_train_real_probabilities_and_outputs(self):
        torch.manual_seed(31)
        train_names = ['event_a_0001', 'event_b_0001']
        val_names = ['event_a_1001', 'event_b_1001']
        front = torch.nn.Linear(1,1).eval().requires_grad_(False)
        head = Readout(dim=8).eval().requires_grad_(False)
        bundle = dict(front=front, heads={'hard':head}, train=train_names, val=val_names,
            device=torch.device('cpu'), summary={}, provenance={'source_head_step':5000},
            manifest={'arguments':{'epsilon':.05,'threshold':.5},'dim':8,'stride':1})
        identity = dict(dim=8, front='synthetic_fixed', pool_mode='hard', read_mode='hard')
        frozen = [copy.deepcopy(m.state_dict()) for m in (front,head)]
        feature = torch.randn(1,256,2,2)
        accesses = []

        def invariant(received):
            self.assertIs(received,bundle)
            for model, state in zip((front,head),frozen):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,state[key],atol=0,rtol=0)

        def loader(dataset,names,train=False,batch=1,workers=0,seed=0):
            names=list(names); accesses.append((train,names))
            self.assertTrue(set(names).issubset(train_names if train else train_names+val_names))
            result=[]
            for start in range(0,len(names),batch):
                current=names[start:start+batch]
                pre=torch.stack([torch.full((3,4,4),.1+int('event_b' in name)*.1) for name in current])
                gt=torch.tensor([[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,255]]).expand(len(current),-1,-1).clone()
                result.append((pre,pre+.1,(gt>0).long(),gt,current))
            return result

        def capture(model,pre,post):
            b=len(pre)
            reference=dict(features={'roi':feature.expand(b,-1,-1,-1)+pre.mean((1,2,3))[:,None,None,None]},
                q=torch.full((b,9,4),1./9),ids=torch.tensor([[0,0,1,1]]).expand(9,-1).clone(),
                tokens=torch.zeros(b,3,8),shape=(2,2),building=torch.tensor([[[[0.,1.],[1.,0.]]]]).expand(b,-1,-1,-1),
                pixel=torch.zeros(b,4,4,4))
            route,_=exp.source.routes(reference,bundle['manifest'],'hard','hard')
            reference['prob']=head(reference['features']['roi'],route,(4,4))
            return reference

        with tempfile.TemporaryDirectory(prefix='label_memory_cpu_') as folder, ExitStack() as stack:
            root=Path(folder)
            for name,value in [('load_source',mock.Mock(return_value=bundle)),('source_identity',mock.Mock(return_value=identity)),
                ('assert_source',invariant),('experiment_hash',mock.Mock(return_value='frozen_old_code')),
                ('loader',loader),('capture',capture),('parse_batch',lambda b,d:b),
                ('seed_everything',lambda seed:torch.manual_seed(seed))]:
                stack.enter_context(mock.patch.object(exp.source,name,value))
            shared=dict(checkpoint='synthetic',run_root='synthetic',dataset='synthetic',train_list='train_set.txt',
                val_list='val_set.txt',source_head='hard',pool_mode='hard',read_mode='hard',workers=0,
                device='cpu',smoke=True,seed=3407)
            with redirect_stdout(io.StringIO()):
                build_bank.run(SimpleNamespace(**shared,output=str(root/'bank'),per_bucket=4))
            bank_bytes=(root/'bank'/'bank.pt').read_bytes()
            for mode in ('audit','train'):
                out=root/mode
                args=SimpleNamespace(**shared,output=str(out),bank=str(root/'bank'),mode=mode,
                    steps=20,eval_every=20,batch=2,accum=8,lr=.01,topk=3,query_chunk=2,temperature=.1,mix_weight=.1)
                with redirect_stdout(io.StringIO()):
                    exp.run(args)
                summary=json.loads((out/'summary.json').read_text())
                self.assertTrue(summary['complete']); self.assertTrue(summary['smoke'])
                self.assertEqual(summary['image_count'],2); self.assertFalse(summary['test_accessed'])
                self.assertTrue(summary['frozen_source_verified'])
                audit=json.loads((out/'audit_final.json').read_text())
                assessment=assess(summary,audit)
                self.assertTrue(assessment['ready'],assessment['missing_evidence'])
                self.assertFalse(assessment['positive_signal'])  # smoke cannot prove efficacy
                self.assertIn('Positive signal: False',render(out))
                candidate='memory' if mode=='train' else 'knn_mix'
                self.assertIn('source_hard__'+candidate,audit['groups']['all']['comparisons'])
                self.assertEqual(summary['final_metrics']['source_hard'],summary['final_metrics']['memory_off' if mode=='train' else 'knn_off'])
                quality=json.loads((out/'retrieval_quality.json').read_text())
                self.assertIn('base_wrong',quality['by_class']['Damaged'])
                evidence=json.loads((out/'label_evidence.json').read_text())
                for counts in evidence['counts'].values():
                    self.assertLessEqual(counts['supported_rescue'],counts['rescued'])
                    self.assertEqual(counts['supported_rescue']+counts['unused_neighbor_opportunity'],
                                     counts['base_wrong_neighbor_correct'])
                if mode=='train':
                    self.assertIn('nomemory__memory',audit['groups']['all']['comparisons'])
                    for name in ('memory','nomemory'):
                        checkpoint=torch.load(out/(name+'_last.pth'),weights_only=False)
                        self.assertEqual(set(checkpoint['head']),{'alpha','log_temperature'})
                        self.assertNotIn('front',checkpoint); self.assertNotIn('optimizer',checkpoint)
                    # A zero mixture is valid; the pipeline must not require positive gains.
                    self.assertIn('VALID negative',summary['note'])
                self.assertEqual((root/'bank'/'bank.pt').read_bytes(),bank_bytes)
                invariant(bundle)
            self.assertIn((True,train_names),accesses)

    def test_probability_decode_does_not_softmax_again(self):
        probability=torch.tensor([[[.01,.02,.96,.01],[.1,.2,.3,.4]]],requires_grad=True)
        route=dict(q=torch.ones(1,1,2),ids=torch.tensor([[[0,1]]]),k=2,shape=(1,2))
        out=exp.decode(probability,route,torch.ones(1,2),(1,2))
        torch.testing.assert_close(out.permute(0,2,3,1).reshape(1,2,4),probability)
        out[:,2].sum().backward()
        self.assertIsNotNone(probability.grad)

    def test_direct_evidence_overlap_and_ignore(self):
        result={}
        exp.add_pixel_complementarity(result,torch.tensor([0,1,2,3,255]),
            torch.tensor([0,0,1,3,0]),torch.tensor([0,1,2,1,2]),torch.tensor([0,1,2,1,1]))
        self.assertEqual(result['all']['rescued'],2)
        self.assertEqual(result['all']['harmed'],1)
        self.assertEqual(result['foreground']['supported_rescue'],2)
        self.assertEqual(result['Destroyed']['aligned_harm'],1)
        self.assertEqual(result['Damaged']['base_wrong_neighbor_correct'],1)

    def test_pure_neighbor_endpoint_has_no_base_cancellation_tie(self):
        from .core import mix_fixed
        base=torch.tensor([[.01,.753,.1,.137]])
        evidence=torch.tensor([[.253,.253,.253,.241]])
        actual=mix_fixed(base,evidence,torch.tensor([True]),1.)
        self.assertTrue(torch.equal(actual,evidence))
        self.assertEqual(actual.argmax(-1).item(),evidence.argmax(-1).item())


if __name__=='__main__':
    unittest.main()
