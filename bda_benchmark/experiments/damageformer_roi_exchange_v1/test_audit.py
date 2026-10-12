import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from .audit import validate_checkpoints, load_heads, evaluate, state_digest, verify_reproduction


class AuditTests(unittest.TestCase):
    def fixture(self):
        manifest=dict(format='ROI_EXCHANGE_V1',test_accessed=False,checkpoint_sha256='f',train_sha256='t',val_sha256='v',frontend_source_hash='s',experiment_hash='e')
        summary=dict(complete=True,test_accessed=False,final_step=10)
        states={n:dict(name=n,step=10,manifest=copy.deepcopy(manifest),head={'weight':torch.ones(1)}) for n in ('hard','exchange')}
        hashes=dict(checkpoint_sha='f',train_sha='t',val_sha='v',frontend_hash='s',experiment_hash='e')
        return manifest,summary,states,hashes

    def test_valid_and_reject_step_manifest_source(self):
        m,s,h,hashes=self.fixture()
        self.assertEqual(validate_checkpoints(m,s,h,**hashes),10)
        for change in ('step','manifest','source','nan','incomplete'):
            m,s,h,hashes=self.fixture()
            if change=='step': h['hard']['step']=9
            if change=='manifest': h['hard']['manifest']['test_accessed']=True
            if change=='source': hashes['frontend_hash']='changed'
            if change=='nan': h['hard']['head']['weight'][0]=float('nan')
            if change=='incomplete': s['complete']=False
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_checkpoints(m,s,h,**hashes)

    def test_strict_head_loading(self):
        cls=lambda dim: torch.nn.Linear(dim,2)
        valid={'hard':dict(head=cls(3).state_dict())}
        loaded=load_heads(valid,3,'cpu',cls)['hard']
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        for weights in ({'weight':torch.ones(2,3)}, {'weight':torch.ones(4,3),'bias':torch.ones(2)}):
            with self.assertRaises(RuntimeError):
                load_heads({'hard':dict(head=weights)},3,'cpu',cls)

    def test_reproduction(self):
        good={'hard':dict(valid_miou=.5,iou=[0.,.5,.5,.5],precision=[None,.5,.5,.5],recall=[None,.5,.5,.5])}
        verify_reproduction(good,copy.deepcopy(good))
        bad=copy.deepcopy(good); bad['hard']['iou'][3]=.4
        with self.assertRaises(ValueError): verify_reproduction(good,bad)
        with self.assertRaises(ValueError): verify_reproduction(good,{})

    def test_frozen_evaluation_has_no_gt_routing(self):
        front=torch.nn.Linear(1,1).eval().requires_grad_(False)
        front.config={'stride':8}
        class Head(torch.nn.Module):
            def forward(self, feature, routing, output_size):
                self.assertion = output_size
                return feature
        head=Head().eval().requires_grad_(False)
        image=torch.ones(1,3,2,2); gt=torch.ones(1,2,2,dtype=torch.long)
        prob=torch.zeros(1,4,2,2); prob[:,1]=1
        reference=dict(pixel=prob,prob=prob,features={'roi':prob},building=torch.ones(1,1,1,1))
        calls=[]
        def route(ref,name,**kwargs):
            self.assertIs(ref,reference)
            self.assertNotIn('gt',ref)
            self.assertEqual(set(kwargs),{'threshold','stride','epsilon'})
            calls.append(name)
            return {}
        before=state_digest(front)
        manifest=dict(format='ROI_EXCHANGE_V1',arguments={'threshold':.5,'epsilon':.05})
        batch=(image,image,None,gt,['a'])
        with patch('bda_benchmark.experiments.damageformer_roi_exchange_v1.audit.parse_batch',side_effect=lambda b,d:b), patch('bda_benchmark.experiments.damageformer_roi_exchange_v1.audit.factorial.capture',return_value=reference):
            result=evaluate(front,{'hard':head},[batch],'cpu',manifest,SimpleNamespace(partition=route))
        self.assertEqual(result['image_count'],1)
        self.assertEqual(calls,['hard'])
        self.assertEqual(before,state_digest(front))
        self.assertFalse(front.training)
        self.assertTrue(all(p.grad is None for p in front.parameters()))


if __name__=='__main__': unittest.main()
