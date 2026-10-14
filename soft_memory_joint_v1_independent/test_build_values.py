"""Synthetic CPU contracts only; no real BRIGHT or CUDA result claims."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch.nn import functional as F

from build_values import run_build, plan_sources, empty_values, label_reference
from common import source_hash
from server_runtime import validate_values, tree_hash
from slots import prepare_slots, slot_labels


def parent_fixture():
    n=4
    return dict(t128=torch.arange(n*128,dtype=torch.float32).reshape(n,128)+1,
                f256=torch.ones(n,256),distribution=F.one_hot(torch.arange(n)%4,4).float(),
                member_ids=torch.arange(n),original_rows=torch.arange(n)+10,images=torch.arange(n),
                events=torch.arange(n),regions=torch.zeros(n,dtype=torch.long),event_names=['e0','e1','e2','e3'],
                eligible=torch.ones(n,dtype=torch.bool),active=torch.ones(n,dtype=torch.bool))


def data_fixture(name='e0_0', target_override=None):
    generator=torch.Generator().manual_seed(91)
    h=w=8; regions=2
    q=torch.full((1,9,h*w),1/9,dtype=torch.float32)
    ids=torch.zeros(9,h*w,dtype=torch.long);ids[4:]=1
    feat=torch.randn(1,256,h,w,generator=generator)
    building=torch.linspace(0,1,h*w).reshape(1,1,h,w)
    target=(torch.arange(h*w).reshape(1,h,w)%4).long()
    target[:,0,:]=255
    if target_override is not None: target=target_override
    slots={rule:prepare_slots(feat,q,ids,regions,(h,w),building,rule) for rule in ('capacity_control','soft')}
    tokens=torch.randn(regions,128,generator=generator)
    pair=dict(t128=tokens,f256=torch.stack([feat.mean((2,3))[0]]*regions),active=torch.ones(regions,dtype=torch.bool),
              mass=torch.tensor([h*w*4/9,h*w*5/9]),q_check={'passed':True})
    logits=torch.randn(1,4,h,w,generator=generator)
    result=dict(q=q,ids=ids,count=regions,shape=(h,w),valid_regions=torch.ones(1,regions,dtype=torch.bool))
    return dict(name=name,event=name.split('_')[0],target=target,result=result,feature=feat,
                pair=pair,native_logits=logits,building=building,slots=slots,
                base_probability_full=logits.softmax(1),native_check={'passed':True,'max_abs':0.0})


class FakeBuildRuntime:
    """Explicit synthetic injection used to exercise persistence/integrity only."""
    def __init__(self,args,output):
        self.args=args;self.output=Path(output);self.bank=parent_fixture()
        self.train_names=['e0_0','e1_0','e2_0','e3_0'];self.val_names=['v0_0','v1_0']
        self.protocol_info=dict(source_protocol_sha256='source-sha',paired_bank_sha256='bank-sha',
            rebuild_protocol_sha256='rebuild-sha',membership_sha256='member-sha',source_protocol='/fake/source.json')
        self.input_ledger={};self.prepare_count=0;self.guard_report=None
    def __enter__(self): return self
    def __exit__(self,*exc):
        self.guard_report=dict(restored=True,failures=[],restore_conflicts=[])
        return False
    def source_batches(self,names): return iter(names)
    def extract(self,name):
        self.prepare_count+=1
        return data_fixture(name)
    def verify(self): return dict(files=True,frozen_models=True,runtime=True,pool_guard=True)


class BuildContracts(unittest.TestCase):
    def args(self,output,limit=0):
        return SimpleNamespace(output=str(output),limit_images=limit,seed=3407)
    def test_full_build_roundtrip_and_all_parent_identity(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'values'
            report=run_build(self.args(out),FakeBuildRuntime)
            self.assertTrue(report['complete']);self.assertFalse(report['smoke'])
            self.assertFalse(report['full_validation_evidence']);self.assertFalse(report['training'])
            value=torch.load(out/'slot_bank.pt',weights_only=False)
            self.assertEqual(value['rules']['soft']['slot_f'].shape,(4,2,256))
            self.assertTrue(value['written'].all())
            validate_values(value,parent_fixture(),smoke=False)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['code_sha256'],source_hash())
            self.assertTrue(manifest['label_reference_passed'])
            self.assertTrue(manifest['roundtrip_exact'])
            self.assertEqual(len((out/'images.jsonl').read_text().splitlines()),4)
    def test_smoke_explicit_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'values'
            result=run_build(self.args(out,4),FakeBuildRuntime)
            self.assertTrue(result['smoke'])
            with self.assertRaises(FileExistsError): run_build(self.args(out,4),FakeBuildRuntime)
    def test_failed_reference_stays_incomplete(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'failed'
            with patch('build_values.label_reference',return_value={'passed':False}):
                with self.assertRaisesRegex(ValueError,'FP64'):
                    run_build(self.args(out),FakeBuildRuntime)
            summary=json.loads((out/'summary.json').read_text())
            self.assertFalse(summary['complete']);self.assertTrue((out/'traceback.txt').is_file())
            self.assertFalse((out/'slot_bank.pt').exists())
    def test_frozen_failure_stops_before_saved_bank(self):
        class Broken(FakeBuildRuntime):
            def verify(self): return {'frozen_models':False}
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'failed'
            with self.assertRaisesRegex(RuntimeError,'integrity'):
                run_build(self.args(out),Broken)
            self.assertFalse((out/'slot_bank.pt').exists())
    def test_incomplete_source_iteration_is_failure(self):
        class Broken(FakeBuildRuntime):
            def source_batches(self,names): return iter(names[:-1])
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'failed'
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                run_build(self.args(out),Broken)
    def test_reference_ignored_area_and_both_rules(self):
        data=data_fixture()
        for rule,slot in data['slots'].items():
            labels,mass=slot_labels(data['target'],slot['route_q'],slot['route_ids'],(8,8),4)
            report=label_reference(data['target'],slot['route_q'],slot['route_ids'],(8,8),4,labels,mass)
            self.assertTrue(report['passed'],report)
            self.assertTrue((mass<slot['slot_mass']).all())
    def test_reference_detects_slot_label_swap(self):
        data=data_fixture()
        slot=data['slots']['soft']
        labels,mass=slot_labels(data['target'],slot['route_q'],slot['route_ids'],(8,8),4)
        bad=labels.clone();bad[...,0]=1;bad[...,1:]=0
        report=label_reference(data['target'],slot['route_q'],slot['route_ids'],(8,8),4,bad,mass)
        self.assertFalse(report['passed'])
    def test_source_plan_cross_event_and_train_only(self):
        bank=parent_fixture();train=['e0_0','e1_0','e2_0','e3_0']
        ids,names=plan_sources(bank,train,['v0'],4)
        self.assertEqual(ids,[0,1,2,3]);self.assertEqual(names,train)
        bank['images'][0]=8
        with self.assertRaisesRegex(ValueError,'Train'): plan_sources(bank,train,['v0'],0)
    def test_source_plan_rejects_single_event_smoke(self):
        bank=parent_fixture();bank['events'].zero_()
        with self.assertRaisesRegex(ValueError,'two support source events'):
            plan_sources(bank,['a','b','c','d'],[],4)
    def test_full_and_smoke_member_written_rules(self):
        bank=parent_fixture();value=empty_values(bank)
        with self.assertRaisesRegex(ValueError,'every original'): validate_values(value,bank,smoke=False)
        with self.assertRaises(ValueError): validate_values(value,bank,smoke=True)
    def test_content_hash_covers_nested_values(self):
        data={'foo':{'x':torch.tensor([1.,2.])}}
        old=tree_hash(data);data['foo']['x'][0]=3
        self.assertNotEqual(tree_hash(data),old)
    def test_native_failure_is_preserved_no_fallback(self):
        class Broken(FakeBuildRuntime):
            def extract(self,name): raise RuntimeError('native scatter_add index assertion')
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'failed'
            with self.assertRaisesRegex(RuntimeError,'native scatter'):
                run_build(self.args(out),Broken)
            self.assertIn('native scatter',json.loads((out/'summary.json').read_text())['error'])


if __name__=='__main__': unittest.main()
