import contextlib
import copy
import gc
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
import weakref
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bright_model_lifetime_v1.pyz'))
sys.path.append(str(HERE.parent/'bright_observer_ab_20261022'))
import test_ab as fixtures
import torch
import life_entry as entry
import life_scoring as scoring
import life_reports as reports
from life_context import CASES,COUNT
from life_observers import execution, python_tensor_inventory, probe_collection, validate_evidence
from evaluation_observer import EvaluationObserver
from evaluate_checkpoint import ARMS, REQUIRED_CHECKS, write_json


class Execution(unittest.TestCase):
    setUp=fixtures.Observers.setUp
    infer=fixtures.Observers.infer

    def context(self,mode):
        self.state=fixtures.training_fixture.CPUState()
        self.observer=EvaluationObserver(self.root,self.position)
        return execution(torch,self.runtime,self.factory,self.ev,self.root,self.position,
                         ['image'],self.state,self.observer,type(self.heads['d']),mode,self.trace)

    def run_unchanged(self,mode):
        with torch.no_grad(): expected=self.infer()
        self.position['index']=-1
        before={a:copy.deepcopy(h.state_dict()) for a,h in self.heads.items()}
        rng=torch.get_rng_state().clone()
        # infer() resets index; instrumented track_evaluation requires -1 on entry.
        def infer():
            self.position.update(index=-1,image=None,active_arm=None,stage='prepare')
            example=self.runtime.prepare(None);support=self.factory.prepare(example,ARMS)
            result={}
            for arm in ARMS:
                self.position['active_arm']=arm
                p=self.factory.forward(self.heads[arm],example,arm,support[arm])['probabilities']
                self.ev._validate_probability(p,example.target,arm);result[arm]=p.clone()
            return result
        prepare=self.runtime.prepare;forward=self.factory.forward
        with torch.no_grad(),self.context(mode):
            if mode=='minimal':
                self.assertIs(self.runtime.prepare,prepare)
                self.assertEqual(self.factory.forward,forward)
                for m in self.front.modules(): self.assertFalse(m._forward_hooks)
            actual=infer()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for a in ARMS:
            self.assertTrue(torch.equal(expected[a],actual[a]))
            for k,v in before[a].items():self.assertTrue(torch.equal(v,self.heads[a].state_dict()[k]))
        s=json.loads((self.root/'lifetime_switch.json').read_text())
        self.assertTrue(s['bindings_restored']);self.assertTrue(s['hooks_restored'])
        self.assertEqual(s['added_success_path_wrappers'],mode=='instrumented')

    def test_minimal_keeps_methods_and_predictions_unchanged(self):self.run_unchanged('minimal')
    def test_instrumented_keeps_predictions_weights_rng_and_restores_methods(self):self.run_unchanged('instrumented')
    def test_minimal_original_route_rejection(self):
        self.example.common['q'].fill_(0)
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'route'),self.context('minimal'):
            self.infer()
    def test_failure_no_cuda_cleanup_and_no_gc(self):
        err=torch.cuda.OutOfMemoryError('synthetic');at=[]
        def fail(*a):at.append(self.cuda.calls);raise err
        self.front[0].forward=fail
        with mock.patch('gc.collect',side_effect=AssertionError('no GC on failure')),torch.no_grad():
            with self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
                with self.context('minimal'):self.infer()
        self.assertIs(caught.exception,err);self.assertEqual(at[0],self.cuda.calls)


class Collection(unittest.TestCase):
    def test_inventory_keeps_no_strong_tensor_references(self):
        x=torch.ones(17);ref=weakref.ref(x)
        refs,summary=python_tensor_inventory(torch,'cpu')
        self.assertIn(id(x),[id(r()) for r in refs])
        del x
        self.assertIsNone(ref())
        self.assertGreaterEqual(summary['python_visible_unique_storage_bytes'],17*4)
    def test_observed_cycle_can_be_collected(self):
        old=gc.isenabled();gc.disable()
        try:
            x=torch.ones(19);ref=weakref.ref(x);cycle=[x];cycle.append(cycle)
            del x,cycle
            refs,_=python_tensor_inventory(torch,'cpu')
            self.assertIsNotNone(ref());gc.collect();self.assertIsNone(ref())
        finally:
            if old:gc.enable()
    def test_collection_uses_no_empty_cache_and_retains_live_tensor_values(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cuda=fixtures.FakeCuda()
            trace=fixtures.MemoryTrace(SimpleNamespace(cuda=cuda,is_grad_enabled=torch.is_grad_enabled),root)
            x=torch.arange(11);expected=x.clone();rng=torch.get_rng_state().clone()
            with mock.patch.object(torch.cuda,'empty_cache',side_effect=AssertionError('no empty_cache')), \
                 mock.patch('life_observers.python_tensor_inventory',return_value=([weakref.ref(x)],{})):
                r=probe_collection(torch,trace,root)
            trace.close()
            self.assertTrue(torch.equal(x,expected));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(r['python_weakrefs_alive_after_gc'],1)
            self.assertEqual(r['optimizer_updates'],0);self.assertFalse(r['historical_failure_resolved'])
    def test_failed_trace_cannot_run_collection(self):
        with mock.patch('gc.collect',side_effect=AssertionError('no collect')):
            with self.assertRaises(ValueError):probe_collection(torch,SimpleNamespace(failed=True),Path('/tmp'))


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.Report.setUpClass()
    def setUp(self):fixtures.Report.setUp(self)
    def infer(self):
        position={}
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No updates')):
            return scoring.evaluate_test(self.runtime,self.f.heads,self.names,self.folder,seed=3407,
                original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,evaluator=fixtures.evaluator,
                batches=lambda:self.runtime.batches(self.names,train=False,seed=3407),step=1500,position=position,mode='minimal')
    def test_no_error_wrapper_needed_for_normal_scoring_and_report(self):
        r=self.infer()
        write_json(self.root/'manifest.json',dict(names=self.names,canonical_test_names=self.names+['later']))
        write_json(self.root/'lifetime_switch.json',dict(mode='minimal',added_success_path_wrappers=False,
            hooks_restored=True,bindings_restored=True,original_validators_preserved=True,tensor_values_modified=False))
        write_json(self.root/'gc_observation.json',dict(gc_after_scoring_only=True,empty_cache_called=False,
            optimizer_updates=0,historical_failure_resolved=False))
        final=reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
            checks={k:True for k in REQUIRED_CHECKS},provenance=dict(case='minimal'),full=False,evidence=None)
        self.assertTrue(final['diagnostic_complete']);self.assertFalse(final['full_test_evidence'])
        self.assertFalse(final['complete'])
    def test_partial_report_cannot_become_full_test(self):
        r=self.infer()
        with self.assertRaises(ValueError):
            reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
                checks={k:True for k in REQUIRED_CHECKS},provenance={},full=True,evidence=None)
    def test_cuda_exception_keeps_primary_and_skips_rng_restore(self):
        err=RuntimeError('CUDA error: device-side assert triggered')
        with mock.patch.object(self.runtime,'prepare',side_effect=err), \
             mock.patch.object(self.state,'restore_rng',side_effect=AssertionError('must not restore')):
            with self.assertRaises(RuntimeError) as caught:self.infer()
        self.assertIs(caught.exception,err)


class Packaging(unittest.TestCase):
    def test_archive_provenance_and_sources(self):
        with zipfile.ZipFile(HERE/'bright_model_lifetime_v1.pyz') as z:
            self.assertIsNone(z.testzip())
            ledger=json.loads(z.read('MODEL_LIFETIME_BASE.json'))['files']
            for n,h in ledger.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h,n)
            for n in ('life_context','life_observers','life_assessment','life_scoring','life_reports','life_entry'):
                self.assertEqual(z.read(n+'.py'),(HERE/(n+'.py')).read_bytes())
    def test_both_cases_memcheck_one_image(self):
        self.assertEqual(COUNT,1);self.assertEqual(CASES,('minimal','instrumented'))
        for c in CASES:
            cmd=entry.command_for(Path('/tmp/job'),c,{'sanitizer':{'path':'/tool'}})
            self.assertEqual(cmd[:5],['/tool','--tool','memcheck','--error-exitcode','86'])
    def test_pair_completion_is_not_full_test_or_historical_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
                 mock.patch.object(entry,'read_request',return_value={'evidence_files':{}}), \
                 mock.patch.object(entry,'require_idle'),mock.patch.object(entry,'supervise',return_value={}), \
                 mock.patch.object(entry,'accept',return_value={'accepted':True}),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.coordinate(root),0)
            s=json.loads((root/'summary.json').read_text())
            self.assertEqual(s['phase'],'model_lifetime_pair_complete');self.assertTrue(s['job_finished'])
            self.assertFalse(s['historical_failure_resolved']);self.assertFalse(s['full_test_evidence'])
            self.assertFalse(s['training_performed'])


if __name__=='__main__':unittest.main()
