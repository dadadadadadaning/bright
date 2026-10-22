import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bright_observer_ab_v1.pyz'))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'],os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import torch
import ab_entry as entry
import ab_scoring as scoring
import ab_reports as reports
from ab_context import CASES, COUNT
from ab_observers import observer_context, validate_observers
from memory_trace import MemoryTrace
from evaluate_checkpoint import ARMS, REQUIRED_CHECKS, write_json
from evaluation_observer import EvaluationObserver, track_evaluation
import test_supports as real
from supports import SupportFactory
import evaluation as evaluator
import test_evaluation as fixtures
import test_resume_integration as training_fixture


class FakeCuda:
    def __init__(self): self.calls=0
    def is_initialized(self): return True
    def memory_stats(self): self.calls+=1; return {}
    def mem_get_info(self): self.calls+=1; return 2000,4000
    def memory_allocated(self): self.calls+=1; return 1000
    def memory_reserved(self): self.calls+=1; return 1500
    def max_memory_allocated(self): self.calls+=1; return 1200
    def max_memory_reserved(self): self.calls+=1; return 1600


class Observers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cuda=FakeCuda()
        self.trace=MemoryTrace(SimpleNamespace(cuda=self.cuda,is_grad_enabled=torch.is_grad_enabled),self.root)
        self.addCleanup(self.trace.close)
        self.runtime,self.example=real.make_fixture();self.example.image='image'
        self.front=torch.nn.Sequential(torch.nn.Identity());self.runtime.bundle={'front':self.front}
        self.position=dict(index=0,image='image',active_arm=None,stage='prepare')
        def prepare(batch): self.front(self.example.common['pixel_features']);return self.example
        self.runtime.prepare=prepare
        self.factory=SupportFactory(self.runtime,3407)
        self.heads={a:real._model().eval() for a in ARMS}
        self.ev=SimpleNamespace(_validate_probability=evaluator._validate_probability)

    def context(self,mode):
        return observer_context(torch,self.runtime,self.factory,self.ev,self.root,self.position,
            ['image'],['image'],mode,'never',heads=self.heads,trace=self.trace)

    def infer(self):
        self.position.update(index=0,image='image',active_arm=None,stage='prepare')
        example=self.runtime.prepare(None);support=self.factory.prepare(example,ARMS)
        result={}
        for arm in ARMS:
            self.position['active_arm']=arm
            p=self.factory.forward(self.heads[arm],example,arm,support[arm])['probabilities']
            self.ev._validate_probability(p,example.target,arm);result[arm]=p.clone()
        return result

    def unchanged(self,mode):
        state=training_fixture.CPUState()
        original=self.factory.forward.__func__
        before=state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()})
        with torch.no_grad(): expected=self.infer()
        rng=state.state_fingerprint(state.capture_rng())
        with torch.no_grad(),self.context(mode): actual=self.infer()
        self.assertEqual(before,state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()}))
        self.assertEqual(rng,state.state_fingerprint(state.capture_rng()))
        self.assertIs(original,self.factory.forward.__func__)
        for arm in ARMS:self.assertTrue(torch.equal(expected[arm],actual[arm]))
        for m in self.front.modules():self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)
        s=json.loads((self.root/'observer_switch.json').read_text())
        self.assertTrue(s['front_hooks_restored']);self.assertEqual(s['extra_route_observers'],mode=='watched')
        if mode=='light':
            self.assertFalse((self.root/'operator_observer_summary.json').exists())
            self.assertEqual(validate_observers(self.root,['image'],mode)['route_snapshot_checks'],'not_performed_in_light_case')
        else:self.assertTrue((self.root/'operator_observer_summary.json').exists())

    def test_light_changes_no_predictions_parameters_rng_or_hooks(self):self.unchanged('light')
    def test_watched_changes_no_predictions_parameters_rng_or_hooks(self):self.unchanged('watched')
    def test_light_preserves_original_invalid_route_rejection(self):
        self.example.common['q'].fill_(0)
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'route'),self.context('light'):
            self.infer()

    def test_failure_does_not_query_cuda_again(self):
        error=torch.cuda.OutOfMemoryError('synthetic');at=[]
        def fail(*args):at.append(self.cuda.calls);raise error
        self.front[0].forward=fail
        with torch.no_grad(),self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
            with self.context('light'):self.infer()
        self.assertIs(caught.exception,error);self.assertEqual(at[0],self.cuda.calls)
        self.assertTrue(json.loads((self.root/'observer_switch.json').read_text())['front_hooks_restored'])


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f=fixtures.EvaluationContracts();self.f.setUp();self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime=self.f.runtime();self.names=['test0'];self.root=self.f.folder.parent
        self.folder=self.root/'test_step_001500';self.state=self.f.legacy.state
    def evaluate(self):
        obs=EvaluationObserver(self.root,dict(index=-1,image=None,active_arm=None,stage='loading'))
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No updates')),track_evaluation(self.runtime,self.f.factory,evaluator,self.names,self.state,obs):
            return scoring.evaluate_test(self.runtime,self.f.heads,self.names,self.folder,seed=3407,
                original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,evaluator=evaluator,
                batches=lambda:self.runtime.batches(self.names,train=False,seed=3407),step=1500)
    def test_one_image_light_report_is_diagnostic_only(self):
        r=self.evaluate()
        write_json(self.root/'manifest.json',dict(names=self.names,canonical_test_names=self.names+['later']))
        write_json(self.root/'observer_switch.json',dict(mode='light',extra_route_observers=False,
            front_hooks_restored=True,route_values_modified=False,original_validators_preserved=True,telemetry_rows=5))
        final=reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
            checks={k:True for k in REQUIRED_CHECKS},provenance=dict(case='light',extra_route_observers=False),full=False,evidence=None)
        self.assertTrue(final['diagnostic_complete']);self.assertFalse(final['complete'])
        self.assertFalse(final['full_test_evidence']);self.assertEqual(final['image_count'],1)
    def test_cannot_promote_to_full_test(self):
        r=self.evaluate()
        with self.assertRaises(ValueError):
            reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
                checks={k:True for k in REQUIRED_CHECKS},provenance={},full=True,evidence=None)


class Packaging(unittest.TestCase):
    def test_both_cases_use_same_sanitizer_command(self):
        root=Path('/tmp/ab');r={'sanitizer':{'path':'/tool'}}
        for case in CASES:
            c=entry.command_for(root,case,r)
            self.assertEqual(c[:5],['/tool','--tool','memcheck','--error-exitcode','86'])
            self.assertEqual(c[-1],case)
        self.assertEqual(COUNT,1)
    def test_inherited_bytes_and_new_source_match(self):
        with zipfile.ZipFile(HERE/'bright_observer_ab_v1.pyz') as z:
            self.assertIsNone(z.testzip())
            ledger=json.loads(z.read('AB_BASE.json'))['files']
            for n,h in ledger.items(): self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h,n)
            for name in ('ab_context','ab_observers','ab_assessment','ab_entry','ab_reports','ab_scoring'):
                self.assertEqual(z.read(name+'.py'),(HERE/(name+'.py')).read_bytes())
    def test_two_successes_do_not_resolve_historical_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
                 mock.patch.object(entry,'read_request',return_value={'evidence_files':{}}), \
                 mock.patch.object(entry,'require_idle'),mock.patch.object(entry,'supervise',return_value={}), \
                 mock.patch.object(entry,'accept',return_value={'accepted':True}),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.coordinate(root),0)
            s=json.loads((root/'summary.json').read_text())
            self.assertEqual(s['phase'],'observer_pair_complete')
            self.assertEqual(set(s['cases']),set(CASES))
            self.assertFalse(s['training_performed']);self.assertFalse(s['full_test_evidence'])
            self.assertFalse(s['historical_failure_resolved'])


if __name__=='__main__':unittest.main()
