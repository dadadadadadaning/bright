"""CPU contracts for the new bridge; these do not validate BRIGHT/CUDA."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from build_values import run_build
from common import digest
from model import JointRegionMemoryClassifier
from server_runtime import (Runtime,FrozenSlotRuntime,BuildingCapture,validate_values,validate_selected,
                            json_hash,VALUE_SCHEMA,LABEL_SCHEMA)
from test_build_values import FakeBuildRuntime,parent_fixture,data_fixture


def mean_pool(values,q,ids,count):
    b,c,h,w=values.shape
    total=torch.zeros(b,count,c);mass=torch.zeros(b,count)
    flat=values.flatten(2).transpose(1,2)
    for k in range(q.shape[1]):
        total.scatter_add_(1,ids[k][None,:,None].expand(b,-1,c),flat*q[:,k,:,None])
        mass.scatter_add_(1,ids[k][None].expand(b,-1),q[:,k])
    return total/torch.where(mass>0,mass,1)[...,None],mass


class QueryRuntime(Runtime):
    def __init__(self,target=None):
        self.args=SimpleNamespace(topk=8,chunk_size=2,key_name='t128')
        self.device=torch.device('cpu');self.bank=parent_fixture();self._train_ids={'e0_0':0}
        self.retrieval_bank=self.bank
        self.select_calls=0;self._data=data_fixture(target_override=target)
        rules={}
        for rule,slot in self._data['slots'].items():
            rules[rule]=dict(slot_f=slot['slot_f'][0,0:1].expand(4,-1,-1).clone(),
                slot_fraction=slot['slot_fraction'][0,0:1].expand(4,-1).clone(),
                slot_labels=torch.full((4,2,4),.25),slot_valid=torch.ones(4,2,dtype=torch.bool))
        self.values={'rules':rules}
        self.deps=SimpleNamespace(mean_pool=mean_pool,select_neighbors=self._select)
    def _select(self,query,bank,**kwargs):
        self.select_calls+=1
        available=(bank['eligible']&(bank['events']!=kwargs['event_id'])&(bank['images']!=kwargs['image_id'])).nonzero().flatten()
        result=torch.full((len(query),kwargs['topk']),-1,dtype=torch.long)
        result[:,:len(available)]=available[None]
        return result,torch.zeros_like(result,dtype=torch.float32)
    def extract(self,batch): return self._data


class RuntimeContracts(unittest.TestCase):
    def test_one_parent_search_and_matched_local_memory_ids(self):
        runtime=QueryRuntime();example=runtime.prepare(None)
        self.assertEqual(runtime.select_calls,1)
        self.assertEqual(set(example.supports),{'a','b','c','d'})
        self.assertTrue(torch.equal(example.supports['a']['support_ids'],example.supports['b']['support_ids']))
        self.assertTrue(torch.equal(example.supports['c']['support_ids'],example.supports['d']['support_ids']))
        self.assertEqual(example.supports['c']['support_f'].shape,(4,2,256))
        self.assertEqual(example.supports['c']['support_t'].shape,(4,128))
        self.assertEqual(example.supports['a']['support_f'].shape,(2,2,256))
    def test_query_gt_never_enters_supports_or_features(self):
        first=QueryRuntime().prepare(None)
        second=QueryRuntime(torch.full((1,8,8),3,dtype=torch.long)).prepare(None)
        for name in first.common: self.assertTrue(torch.equal(first.common[name],second.common[name]),name)
        for arm in first.supports:
            for name in first.supports[arm]:
                self.assertTrue(torch.equal(first.supports[arm][name],second.supports[arm][name]),arm+'/'+name)
    def test_all_arms_connect_to_actual_model_and_zero_init(self):
        example=QueryRuntime().prepare(None)
        head=JointRegionMemoryClassifier(chunk_size=2)
        for arm in example.supports:
            result=head(**example.common,**example.supports[arm])
            probability=result['probabilities']
            self.assertTrue(torch.allclose(probability,example.common['base_probability_full'],atol=1e-6,rtol=1e-6),arm)
    def test_same_event_and_same_image_rechecked_before_gather(self):
        bank=parent_fixture()
        with self.assertRaisesRegex(ValueError,'same-event'):
            validate_selected(torch.tensor([[0,1]]),bank,event_id=0,image_id=-1)
        with self.assertRaisesRegex(ValueError,'same-event or same-image'):
            validate_selected(torch.tensor([[0,1]]),bank,event_id=-1,image_id=1)
    def test_ineligible_and_bad_ids_rejected(self):
        bank=parent_fixture();bank['eligible'][2]=False
        with self.assertRaisesRegex(ValueError,'unavailable'):
            validate_selected(torch.tensor([[2]]),bank,event_id=-1,image_id=-1)
        for ids in (torch.tensor([[-2]]),torch.tensor([[4]])):
            with self.assertRaisesRegex(ValueError,'out of range'):
                validate_selected(ids,bank,event_id=-1,image_id=-1)
    def test_parent_padding_does_not_expand_bank_table(self):
        example=QueryRuntime().prepare(None)
        self.assertEqual(example.supports['c']['support_ids'].shape,(2,8))
        self.assertEqual(example.supports['c']['support_f'].shape,(4,2,256))
        self.assertTrue((example.supports['c']['support_ids'][:,3:]==-1).all())
    def test_local_labels_are_same_route_native_predictions(self):
        runtime=QueryRuntime();example=runtime.prepare(None)
        from slots import slot_pool
        for arm,rule in (('a','capacity_control'),('b','soft')):
            slot=runtime._data['slots'][rule]
            expected,_=slot_pool(runtime._data['native_logits'].softmax(1),slot['route_q'],slot['route_ids'],slot['slot_count'])
            self.assertTrue(torch.equal(example.supports[arm]['support_labels'],expected[0]))
    def test_building_capture_exact_partition_input(self):
        class Partition(nn.Module):
            def forward(self,feature,b): return feature+b
        front=SimpleNamespace(partition=Partition())
        capture=BuildingCapture(front);feature=torch.zeros(1,1,2,2);b=torch.rand(1,1,2,2)
        front.partition(feature,b)
        actual=capture.take()
        self.assertTrue(torch.equal(actual,b));self.assertEqual(actual.data_ptr(),b.data_ptr())
        capture.close()
    def test_building_capture_zero_and_double_calls_fail(self):
        class Partition(nn.Module):
            def forward(self,feature,b): return feature
        front=SimpleNamespace(partition=Partition());capture=BuildingCapture(front)
        with self.assertRaises(RuntimeError): capture.take()
        b=torch.rand(1,1,2,2)
        front.partition(b,b);front.partition(b,b)
        with self.assertRaises(RuntimeError): capture.take()
        capture.close()


class ValueLoaderContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.values=self.root/'values'
        run_build(SimpleNamespace(output=str(self.values),limit_images=0),FakeBuildRuntime)
    def tearDown(self): self.temp.cleanup()
    def _args(self,**kwargs):
        data=dict(values=str(self.values),key_name='t128',mode='smoke',smoke=True,allow_smoke_values=False)
        data.update(kwargs);return SimpleNamespace(**data)
    def _base_enter(self,runtime):
        runtime.device=torch.device('cpu');runtime.bank=parent_fixture()
        runtime.protocol_info=FakeBuildRuntime(None,self.root).protocol_info
        runtime.input_ledger={}
        return runtime
    def _load(self,args=None):
        runtime=Runtime(args or self._args(),self.root/'runtime')
        with patch.object(FrozenSlotRuntime,'__enter__',lambda runtime:self._base_enter(runtime)),patch.object(FrozenSlotRuntime,'close',lambda *a:None):
            return runtime.__enter__()
    def _alter_manifest(self,key,value):
        p=self.values/'manifest.json';d=json.loads(p.read_text());d[key]=value;p.write_text(json.dumps(d))
    def test_valid_formal_values_load(self):
        runtime=self._load()
        self.assertEqual(runtime.values['rules']['soft']['slot_f'].shape,(4,2,256))
        self.assertTrue(runtime.retrieval_bank['eligible'].all())
    def test_wrong_source_protocol_rejected(self):
        self._alter_manifest('source_protocol_sha256','wrong')
        with self.assertRaisesRegex(ValueError,'provenance mismatch'): self._load()
    def test_changed_bank_file_rejected(self):
        with (self.values/'slot_bank.pt').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'): self._load()
    def test_code_change_rejected(self):
        self._alter_manifest('code_sha256','wrong')
        with self.assertRaisesRegex(ValueError,'code_sha256'): self._load()
    def test_smoke_requires_both_explicit_flags(self):
        self._alter_manifest('smoke',True)
        with self.assertRaisesRegex(ValueError,'allow-smoke-values'): self._load()
        with self.assertRaisesRegex(ValueError,'Formal training'):
            self._load(self._args(allow_smoke_values=True,smoke=False))
        runtime=self._load(self._args(allow_smoke_values=True))
        self.assertTrue(runtime.protocol_info['child_values_smoke'])
    def test_original_identity_change_rejected_even_with_updated_file_sha(self):
        value=torch.load(self.values/'slot_bank.pt',weights_only=False)
        value['identity']['regions'][0]=1
        torch.save(value,self.values/'slot_bank.pt')
        path=self.values/'manifest.json';manifest=json.loads(path.read_text())
        manifest['file_sha256']['slot_bank.pt']=digest(self.values/'slot_bank.pt');path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'identity mismatch'): self._load()
    def test_valid_label_mass_cannot_exceed_feature_mass(self):
        value=torch.load(self.values/'slot_bank.pt',weights_only=False)
        value['rules']['soft']['label_mass'][0]=1e8
        with self.assertRaisesRegex(ValueError,'exceeds assignment mass'):
            validate_values(value,parent_fixture(),smoke=False)
    def test_wrong_value_schema_rejected(self):
        self._alter_manifest('value_schema_sha256','wrong')
        with self.assertRaisesRegex(ValueError,'value_schema'): self._load()
    def test_source_image_changed_since_build_rejected(self):
        image=self.root/'image';image.write_bytes(b'old')
        fingerprint=self.values/'input_fingerprints.json';fingerprint.write_text(json.dumps({str(image):digest(image)}))
        path=self.values/'manifest.json';manifest=json.loads(path.read_text())
        manifest['file_sha256']['input_fingerprints.json']=digest(fingerprint);path.write_text(json.dumps(manifest))
        image.write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'source image changed'): self._load()


if __name__=='__main__': unittest.main()
