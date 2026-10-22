"""CPU contracts with original model/support fixtures; no remote CUDA claim."""
import contextlib
import copy
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bright_test_sweep_v1.pyz'))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'],os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import numpy as np
import torch
import sweep_entry as entry
import sweep_reports as reports
import sweep_scoring as scoring
import sweep_context as contexts
from sweep_routes import StageRoutes
from evaluate_checkpoint import ARMS,REQUIRED_CHECKS,write_json
from grid_probe import sha256
from evaluation_observer import EvaluationObserver,track_evaluation
from passive_routes import PassiveRoutes
import test_supports as real
from supports import SupportFactory
import evaluation as evaluator
import test_evaluation as fixtures
import test_resume_integration as training_fixture


class Boundaries(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.runtime,self.example=real.make_fixture();self.example.image='target'
        self.runtime.prepare=lambda batch:self.example
        self.factory=SupportFactory(self.runtime,3407)
        self.heads={a:real._model().eval() for a in ARMS}
        self.position=dict(index=-1,image='target',active_arm=None,stage='loading')
        self.evaluator=SimpleNamespace(_validate_probability=evaluator._validate_probability)
    def watch(self):
        return StageRoutes(torch,self.runtime,self.factory,self.evaluator,self.root,self.position,
                           ['target'],['target'],'watched','target',heads=self.heads)
    def forward(self):
        e=self.runtime.prepare(None);s=self.factory.prepare(e,ARMS);result={}
        for a in ARMS:
            p=self.factory.forward(self.heads[a],e,a,s[a])['probabilities']
            self.evaluator._validate_probability(p,e.target,a);result[a]=p.detach().clone()
        return result
    def test_original_predictions_unchanged_all_hooks_restored(self):
        state=training_fixture.CPUState();before=state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()})
        scope=real.original_model.JointRegionMemoryClassifier.forward.__globals__
        unpool,F=scope['sparse_unpool'],scope['F']
        with torch.no_grad():expected=self.forward()
        with torch.no_grad(),PassiveRoutes(torch,real.original_model.JointRegionMemoryClassifier,self.root,self.position),self.watch():
            got=self.forward()
        for arm in ARMS:self.assertTrue(torch.equal(got[arm],expected[arm]))
        self.assertEqual(before,state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()}))
        self.assertIs(scope['sparse_unpool'],unpool);self.assertIs(scope['F'],F)
        record=json.loads((self.root/'operator_observer_summary.json').read_text())
        self.assertGreater(record['boundary_calls'],6);self.assertTrue(record['bindings_restored'])
    def test_corruption_localized_to_reader_interval(self):
        original=self.heads['d']._read
        def bad(*args,**kw):
            result=original(*args,**kw)
            self.example.supports['d']['route_q'][0,0,0]*=.5
            return result
        self.heads['d']._read=bad
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'Frozen routing'):
            with self.watch():self.forward()
        window=json.loads((self.root/'operator_window.json').read_text())
        self.assertEqual(window['previous']['position']['watch_stage'],'d._read_before')
        self.assertEqual(window['current']['position']['watch_stage'],'d._read_after')
        self.assertEqual(window['current']['position']['operator'],'d._read')
        data=torch.load(self.root/'operator_window_cpu.pt',weights_only=False)
        self.assertFalse(torch.equal(data['before']['arm.d.route_q'],data['after']['arm.d.route_q']))
    def test_cuda_like_error_preserves_primary_exception_and_restores_hooks(self):
        error=RuntimeError('synthetic device-side assert')
        self.heads['d']._read=lambda *a,**k:(_ for _ in ()).throw(error)
        scope=real.original_model.JointRegionMemoryClassifier.forward.__globals__;original=scope['sparse_unpool']
        with torch.no_grad(),self.assertRaises(RuntimeError) as caught:
            with self.watch():self.forward()
        self.assertIs(caught.exception,error);self.assertIs(scope['sparse_unpool'],original)
        self.assertFalse((self.root/'route_change_cpu.pt').exists())


class FiveStepScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f=fixtures.EvaluationContracts();self.f.setUp();self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime=self.f.runtime();self.names=['test%04d'%i for i in range(878)]
        self.root=self.f.folder.parent;self.state=self.f.legacy.state
    def infer(self,step,names=None):
        names=self.names if names is None else names
        root=self.root/('case%s'%step);root.mkdir();folder=root/('test_step_%06d'%step)
        observer=EvaluationObserver(root,dict(index=-1,image=None,active_arm=None,stage='loading'))
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No Test updates')), \
             track_evaluation(self.runtime,self.f.factory,evaluator,names,self.state,observer):
            r=scoring.evaluate_test(self.runtime,self.f.heads,names,folder,seed=3407,
                original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,evaluator=evaluator,
                batches=lambda:self.runtime.batches(names,train=False,seed=3407),step=step)
        return root,folder,r
    def test_all_five_steps_complete878_with_correct_metadata_and_identical_reference(self):
        rng=self.state.state_fingerprint(self.state.capture_rng());expected=None;records={}
        for step in contexts.STEPS:
            root,folder,r=self.infer(step)
            self.assertEqual((r['step'],r['image_count']),(step,878))
            reports.verify_report(folder/'result.json',r,self.names,step)
            if expected is None:expected=r['metrics']
            self.assertEqual(r['metrics'],expected)
            write_json(root/'manifest.json',dict(names=self.names,protocol_info={'fixture':True}))
            write_json(root/'operator_observer_summary.json',dict(boundary_calls=1,bindings_restored=True))
            final=reports.finish_report(folder/'result.json',r,output=root,names=self.names,evidence=None,
                checks={k:True for k in REQUIRED_CHECKS},provenance=dict(checkpoint_step=step,protocol_info={'fixture':True}),full=True)
            self.assertTrue(final['full_test_evidence']);self.assertFalse(final['test_used_for_selection'])
            records['test%04d'%step]=dict(accepted=True,result=str(folder/'result.json'),result_sha256=sha256(folder/'result.json'))
        self.assertEqual(rng,self.state.state_fingerprint(self.state.capture_rng()))
        combined=reports.export(self.root,dict(cases=records))
        self.assertEqual(combined['complete_steps'],list(contexts.STEPS));self.assertEqual(len(combined['metrics']),35)
        self.assertTrue(combined['frozen_grid_confusions_identical'])
        self.assertNotIn('best_step',combined)
        records['test1500']['accepted']=False
        combined=reports.export(self.root,dict(cases=records))
        self.assertEqual(combined['incomplete_steps'],[1500]);self.assertEqual(len(combined['metrics']),28)
    def test_partial_report_and_wrong_step_rejected(self):
        root,folder,r=self.infer(500,self.names[:2])
        with self.assertRaisesRegex(ValueError,'identity'):reports.verify_report(folder/'result.json',r,self.names,500)
        with self.assertRaisesRegex(ValueError,'identity'):reports.verify_report(folder/'result.json',r,self.names,1000)
    def test_changed_journal_and_forged_metrics_rejected(self):
        root,folder,r=self.infer(1000)
        r['metrics']['d']['miou4']+=.01
        with self.assertRaisesRegex(ValueError,'formula'):reports.verify_report(folder/'result.json',r,self.names,1000)
        r['metrics']['d']['miou4']-=.01
        with (folder/'images.jsonl').open('a') as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'journal changed'):reports.verify_report(folder/'result.json',r,self.names,1000)
    def test_numeric_definitions_differ_for_four_and_three_classes(self):
        values=reports.numeric([[100,0,0,0],[0,10,10,0],[0,0,10,0],[0,0,0,10]])
        self.assertAlmostEqual(values['miou4_pct'],75)
        self.assertAlmostEqual(values['foreground_miou3_pct'],200/3)
        self.assertTrue(all(v is None for v in reports.numeric(np.zeros((4,4),dtype=np.int64)).values()))


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.f=training_fixture.ResumeIntegrationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
    def test_original500_and1000_state_formats_load_without_updates(self):
        for step in (500,1000):
            saved=copy.deepcopy(self.f.saved);saved['step']=step;saved['test_accessed']=False
            for opt in saved['optimizers'].values():
                for state in opt['state'].values():state['step'].fill_(step)
            path=self.f.base/('checkpoint_%06d.pt'%step);torch.save(saved,path)
            meta=dict(step=step,sha256=sha256(path),roundtrip_verified=True,
                head_hashes={a:self.f.state.state_fingerprint(h) for a,h in saved['heads'].items()})
            write_json(path.with_suffix('.pt.json'),meta)
            c=dict(parent=dict(path=str(path),step=step,kind='original',sha256=meta['sha256']),
                   plan=self.f.plan,manifest=self.f.manifest)
            result,_=contexts.load_selected(torch,self.f.state,c)
            self.assertEqual(result['step'],step)
            c['parent']['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'changed'):contexts.load_selected(torch,self.f.state,c)
    def test_resumed_sidecar_rejects_different_journal_prefix(self):
        p=self.f.base/'run/checkpoints/checkpoint_001500.pt';p.parent.mkdir(parents=True);p.write_bytes(b'fixture')
        c=dict(plan={},manifest={},lineage={},parent_journal=self.f.base/'final.jsonl')
        meta=dict(step=1500,sha256=sha256(p),roundtrip_verified=True,head_hashes={a:'hash' for a in ARMS})
        write_json(p.with_suffix('.pt.json'),meta)
        with mock.patch.object(contexts.checkpoint_io,'journal_prefix',side_effect=[{'hash':'final'},{'hash':'other'}]):
            with self.assertRaisesRegex(ValueError,'trajectory'):contexts.validate_sidecar(p,1500,'resume',c)


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def run_cases(self,outcomes):
        calls=[]
        def run(root,case,budget,stop):calls.append(case);return dict(outcomes.get(case,dict(returncode=0,supervisor_reason=None)))
        request=dict(evidence_files={},case_timeout_seconds=60,total_timeout_seconds=300)
        with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
             mock.patch.object(entry,'read_request',return_value=request),mock.patch.object(entry,'supervise',side_effect=run), \
             mock.patch.object(entry,'accept_case',return_value={}),mock.patch.object(entry,'export',return_value={}),contextlib.redirect_stdout(io.StringIO()):
            code=entry.coordinate(self.root)
        return code,calls,json.loads((self.root/'summary.json').read_text())
    def test_failure_does_not_retry_or_erase_other_steps(self):
        code,calls,result=self.run_cases({'test1500':dict(returncode=2,supervisor_reason=None)})
        self.assertEqual(code,2);self.assertEqual(calls,list(entry.CASES))
        self.assertTrue(result['cases']['test2500']['accepted']);self.assertFalse(result['cases']['test1500']['accepted'])
        self.assertFalse(result['complete']);self.assertTrue(result['job_finished'])
    def test_timeout_does_not_accept_stale_success(self):
        _,calls,result=self.run_cases({'test0500':dict(returncode=0,supervisor_reason='time_budget_exceeded')})
        self.assertFalse(result['cases']['test0500']['accepted']);self.assertEqual(len(calls),5)
    def test_configuration_failure_aborts(self):
        _,calls,result=self.run_cases({'test0500':dict(returncode=3,supervisor_reason=None)})
        self.assertEqual(calls,['test0500']);self.assertEqual(result['phase'],'aborted')
    def test_real_child_timeout_keeps_log(self):
        (self.root/'runner.pyz').write_text('import time\nprint("started",flush=True)\ntime.sleep(30)\n')
        with contextlib.redirect_stdout(io.StringIO()):r=entry.supervise(self.root,'test0500',.05,{'requested':False})
        self.assertEqual(r['supervisor_reason'],'time_budget_exceeded');self.assertNotEqual(r['returncode'],0)


if __name__=='__main__':unittest.main()
