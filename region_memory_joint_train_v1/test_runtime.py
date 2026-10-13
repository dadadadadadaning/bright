"""CPU contracts for routing/support binding; no BRIGHT data or CUDA claim."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import json

import torch

try:
    from . import server_runtime as sr
    from .model import sparse_pool
except ImportError:
    import server_runtime as sr
    from model import sparse_pool


class LocalSupportTests(unittest.TestCase):
    def test_self_included_and_region_id_tie(self):
        points = torch.tensor([[.5,.5],[.5,.0],[.5,1.],[.0,.5]])
        selected = sr.local_support_ids(points, torch.ones(4, dtype=torch.bool), 3)
        self.assertEqual(selected[0].tolist(), [0,1,2])
        self.assertTrue(bool((selected[:,0] == torch.arange(4)).all()))

    def test_inactive_and_padding(self):
        points = torch.tensor([[0.,0.],[0.,1.],[1.,0.]])
        selected = sr.local_support_ids(points, torch.tensor([True,False,True]), 4)
        self.assertEqual(selected.tolist(), [[0,2,-1,-1],[-1,-1,-1,-1],[2,0,-1,-1]])

    def test_single_region_has_only_self_support(self):
        selected = sr.local_support_ids(torch.zeros(1,2), torch.ones(1,dtype=torch.bool), 8)
        self.assertEqual(selected.tolist(), [[0,-1,-1,-1,-1,-1,-1,-1]])

    def test_invalid_centroid_fails(self):
        with self.assertRaises(ValueError):
            sr.local_support_ids(torch.tensor([[float('nan'),0.]]), torch.ones(1,dtype=torch.bool), 1)


class GatherTests(unittest.TestCase):
    def setUp(self):
        self.f = torch.arange(3*256,dtype=torch.float32).reshape(3,256)
        self.t = torch.arange(3*128,dtype=torch.float32).reshape(3,128)
        self.y = torch.eye(4,dtype=torch.float32)[:3]

    def test_same_row_binding_and_zero_padding(self):
        result = sr.gather_support(torch.tensor([[2,0,-1]]), self.f, self.t, self.y)
        self.assertTrue(torch.equal(result['support_f'][0,0,0],self.f[2]))
        self.assertTrue(torch.equal(result['support_t'][0,0,0],self.t[2]))
        self.assertTrue(torch.equal(result['support_labels'][0,0,0],self.y[2]))
        self.assertEqual(result['support_valid'].tolist(),[[[True,True,False]]])
        self.assertEqual(float(result['support_f'][0,0,2].abs().sum()),0)

    def test_exclusion_is_rechecked(self):
        for kwargs in (dict(events=torch.tensor([0,1,2]),event_id=1),
                       dict(images=torch.tensor([10,20,30]),image_id=20),
                       dict(eligible=torch.tensor([True,False,True]))):
            with self.assertRaises(ValueError):
                sr.gather_support(torch.tensor([[1]]), self.f, self.t, self.y, **kwargs)

    def test_heldout_sentinel_does_not_exclude_train(self):
        result=sr.gather_support(torch.tensor([[0]]),self.f,self.t,self.y,
                                 images=torch.tensor([0,1,2]),image_id=-1)
        self.assertTrue(result['support_valid'].item())

    def test_bad_negative_and_outside_ids(self):
        for ids in (-2,3):
            with self.assertRaises(ValueError):
                sr.gather_support(torch.tensor([[ids]]),self.f,self.t,self.y)

    def test_invalid_padded_nan_is_not_read(self):
        self.f[0]=float('nan')
        result=sr.gather_support(torch.tensor([[-1,1]]),self.f,self.t,self.y)
        self.assertTrue(torch.isfinite(result['support_f']).all())
        with self.assertRaises(ValueError):
            sr.gather_support(torch.tensor([[0]]),self.f,self.t,self.y)

    def test_selected_bad_label_rejected(self):
        self.y[2]=0
        with self.assertRaises(ValueError):
            sr.gather_support(torch.tensor([[2]]),self.f,self.t,self.y)


class RuntimeBridgeTests(unittest.TestCase):
    def make_runtime(self, directory):
        torch.manual_seed(11)
        rt=sr.Runtime(SimpleNamespace(repo=directory,workers=0,seed=3407,device='cpu',
                                     topk=8,chunk_size=3,v3_root=directory),directory)
        rt.device=torch.device('cpu')
        rt._train_ids={'event-a_0':0,'event-b_0':1,'event-c_0':2,'event-d_0':3}
        rt.train_names=list(rt._train_ids)
        rt.val_names=['event-a_9','event-b_9']
        rt._val_set=set(rt.val_names)
        source=Path(directory)/'source';source.write_bytes(b'unchanged')
        rt.bundle={'args':SimpleNamespace(dataset=directory)}
        n=4
        rt.bank={'f256':torch.randn(4,256),'t128':torch.randn(4,128),
                 'distribution':torch.eye(4),'eligible':torch.ones(4,dtype=torch.bool),
                 'events':torch.arange(4),'images':torch.arange(4),
                 'event_names':['event-a','event-b','event-c','event-d']}
        q=torch.ones(1,1,4)
        ids=torch.tensor([[0,1,2,3]])
        feature=torch.randn(1,256,2,2)
        logits=torch.randn(1,4,2,2)
        tokens=torch.randn(4,128)
        result={'q':q,'ids':ids,'count':n,'shape':(2,2)}
        calls={'front':0,'pair':0,'retrieve':[]}
        def capture(pre,post):
            calls['front']+=1
            return result,feature
        def extract_pair(value,feat):
            calls['pair']+=1
            self.assertIs(value,result)
            self.assertIs(feat,feature)
            means,mass=sparse_pool(feature.flatten(2),q,ids,n)
            return {'t128':tokens,'f256':means[0],'mass':mass[0],
                    'active':torch.tensor([True,True,True,False]),'q_check':{'passed':True}}
        def mean_pool(value,q,ids,count):
            return sparse_pool(value.flatten(2),q,ids,count)
        def select(query,bank,*,key_name,event_id,image_id,topk,policy,chunk_size):
            calls['retrieve'].append((key_name,event_id,image_id,policy,chunk_size))
            candidates=((bank['events']!=event_id)&(bank['images']!=image_id)).nonzero().flatten()
            output=torch.full((len(query),topk),-1,dtype=torch.long)
            output[:,:len(candidates)]=candidates
            return output,torch.zeros_like(output,dtype=torch.float32)
        rt.deps=SimpleNamespace(parse_batch=lambda batch,device:batch,
            sample_paths=lambda dataset,name:{'pre':source,'post':source,'label':source},
            native_grid_probq=lambda *args:(logits.softmax(1),{'passed':True,'max_abs':0.}),
            extract_pair=extract_pair,mean_pool=mean_pool,select_neighbors=select,
            event=lambda name:name.rsplit('_',1)[0])
        rt.capture=capture
        rt.native_capture=SimpleNamespace(clear=lambda:None,take=lambda:logits)
        rt.guard=SimpleNamespace(set_image=lambda name:None)
        return rt,calls

    def test_prepare_uses_one_pass_no_gt_in_keys_or_current_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            rt,calls=self.make_runtime(directory)
            batch=(torch.zeros(1,3,2,2),torch.zeros(1,1,2,2),torch.zeros(1,2,2,dtype=torch.long),torch.zeros(1,2,2,dtype=torch.long),['event-a_0'])
            first=rt.prepare(batch)
            second=rt.prepare((*batch[:3],torch.full_like(batch[3],3),batch[4]))
            self.assertEqual(calls['front'],2)
            self.assertEqual(calls['pair'],2)
            self.assertEqual(calls['retrieve'][0],('t128',0,0,'global',3))
            self.assertEqual(calls['retrieve'][1],('f256',0,0,'global',3))
            for arm in ('r0','rt','rf'):
                self.assertTrue(torch.equal(first.supports[arm]['support_labels'],second.supports[arm]['support_labels']))
                self.assertFalse(first.supports[arm]['support_valid'][0,3].any())
            for key in first.common:
                self.assertTrue(torch.equal(first.common[key],second.common[key]))
            self.assertEqual(set(first.common), {'pixel_features','region_f','region_t','q','ids','base_probability_full'})
            self.assertFalse(first.common['pixel_features'].requires_grad)
            self.assertEqual(len(rt.input_ledger),1)

    def test_raw_invalid_labels_fail_before_front_and_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            rt,calls=self.make_runtime(directory)
            for bad in (256., 1.5, float('nan'), -1.):
                target=torch.zeros(1,2,2);target[0,0,0]=bad
                batch=(torch.zeros(1,3,2,2),torch.zeros(1,1,2,2),torch.zeros(1,2,2),target,['event-a_0'])
                with self.assertRaises(ValueError):
                    rt.prepare(batch)
            self.assertEqual(calls['front'],0)
            self.assertTrue((Path(directory)/'label_failure.json').is_file())

    def test_val_uses_verified_absent_image_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            rt,calls=self.make_runtime(directory)
            batch=(torch.zeros(1,3,2,2),torch.zeros(1,1,2,2),torch.zeros(1,2,2,dtype=torch.long),torch.zeros(1,2,2,dtype=torch.long),['event-b_9'])
            example=rt.prepare(batch)
            self.assertEqual(example.diagnostics['image_id'],-1)
            self.assertEqual(calls['retrieve'][0][1:3],(1,-1))
            with self.assertRaises(ValueError):
                rt.prepare((*batch[:4],['not-in-split']))

    def test_smoke_chooses_distinct_legal_events_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            rt,_=self.make_runtime(directory)
            train,val=rt.smoke_names()
            self.assertEqual(train,rt.train_names)
            self.assertEqual(val,rt.val_names)
            rt.bank['eligible'].zero_()
            with self.assertRaises(ValueError):
                rt.smoke_names()

    def test_source_protocol_resolves_source_ledger_not_key_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source_e0.json';source.write_text('{}')
            (root/'pipeline_state.json').write_text(json.dumps({'protocol':str(source),'complete':False}))
            (root/'keys').mkdir();(root/'keys'/'protocol.json').write_text('{"format":"new_bank"}')
            args=SimpleNamespace(v3_root=directory,protocol=None)
            self.assertEqual(sr.source_protocol(args),source.resolve())

    def test_verify_rejects_in_memory_bank_change(self):
        with tempfile.TemporaryDirectory() as directory:
            rt,_=self.make_runtime(directory)
            rt.deps.shared=SimpleNamespace(verify_all=lambda *args:{'frozen':True})
            rt.own_before='source'
            rt.extra_ledger={}
            rt.bank_before=sr._bank_fingerprint(rt.bank)
            rt.guard.failures=[];rt.guard.restore_conflicts=[]
            self.assertTrue(all(rt.verify().values()))
            rt.bank['distribution'][0]=torch.tensor([0.,1.,0.,0.])
            with self.assertRaises(RuntimeError):
                rt.verify()

    def test_close_restores_guard_and_closes_both_hooks(self):
        rt=sr.Runtime(SimpleNamespace(),'.')
        closed=[]
        class Guard:
            _entered=True
            restored=False
            def __exit__(self,*args):
                self.restored=True
            def report(self):
                return {'restored':self.restored,'restore_conflicts':[]}
        rt.guard=Guard()
        rt.native_capture=SimpleNamespace(close=lambda:closed.append('native'))
        rt.capture=SimpleNamespace(close=lambda:closed.append('feature'))
        rt.__exit__(None,None,None)
        self.assertEqual(closed,['native','feature'])
        self.assertTrue(rt.guard_report['restored'])
        rt.close()
        self.assertEqual(len(closed),2)

    def test_cleanup_preserves_original_error_and_closes_other_hook(self):
        rt=sr.Runtime(SimpleNamespace(),'.')
        closed=[]
        def fail():
            raise OSError('cleanup disk full')
        rt.native_capture=SimpleNamespace(close=fail)
        rt.capture=SimpleNamespace(close=lambda:closed.append('feature'))
        original=ValueError('original GPU failure')
        rt.close(ValueError,original,None)
        self.assertEqual(closed,['feature'])
        self.assertIn('cleanup_failures',rt.guard_report)

    def test_bank_fingerprint_detects_value_and_identity_changes(self):
        bank={'f256':torch.ones(2,256),'images':torch.tensor([0,1]),'split':'train'}
        before=sr._bank_fingerprint(bank)
        bank['f256'][0,0]=2
        self.assertNotEqual(before,sr._bank_fingerprint(bank))


if __name__=='__main__':
    unittest.main()
