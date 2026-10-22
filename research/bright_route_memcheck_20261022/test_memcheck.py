"""CPU model/observer contracts and real child supervision; no CUDA claim."""
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'bright_route_memcheck_v1.pyz'))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'], os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import torch
import memcheck_entry as entry
import memcheck_context as contexts
import memcheck_reports as reports
import memcheck_scoring as scoring
from memcheck_routes import CorroboratedRoutes
from evaluate_checkpoint import ARMS, REQUIRED_CHECKS, write_json
from evaluation_observer import EvaluationObserver, track_evaluation
from grid_probe import sha256
import test_supports as real
from supports import SupportFactory
import evaluation as evaluator
import test_evaluation as fixtures
import test_resume_integration as training_fixture


class ObserverContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.runtime,self.example=real.make_fixture(); self.example.image='image'
        self.runtime.prepare=lambda batch:self.example
        self.factory=SupportFactory(self.runtime,3407)
        self.heads={a:real._model().eval() for a in ARMS}
        self.position=dict(index=-1,image='image',active_arm=None,stage='loading')
        self.evaluator=SimpleNamespace(_validate_probability=evaluator._validate_probability)
        self.w=CorroboratedRoutes(torch,self.runtime,self.factory,self.evaluator,self.root,self.position,
            ['image'],['image'],'watched',contexts.TARGET,heads=self.heads)
    def infer(self):
        e=self.runtime.prepare(None); support=self.factory.prepare(e,ARMS); result={}
        for arm in ARMS:
            p=self.factory.forward(self.heads[arm],e,arm,support[arm])['probabilities']
            self.evaluator._validate_probability(p,e.target,arm); result[arm]=p.detach().clone()
        return result
    def report(self): return json.loads((self.root/'parent_ids_corroboration.json').read_text())
    def test_real_six_arm_outputs_and_parameters_unchanged(self):
        state=training_fixture.CPUState()
        before=state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()})
        scope=real.original_model.JointRegionMemoryClassifier.forward.__globals__
        unpool,F=scope['sparse_unpool'],scope['F']
        with torch.no_grad(): expected=self.infer()
        with torch.no_grad(), self.w: got=self.infer()
        self.assertEqual(before,state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()}))
        for arm in ARMS:self.assertTrue(torch.equal(expected[arm],got[arm]))
        self.assertIs(scope['sparse_unpool'],unpool); self.assertIs(scope['F'],F)
        self.assertFalse(self.w.confirmed); self.assertGreater(self.w.device_comparisons,20)
        meta=json.loads((self.root/'operator_observer_summary.json').read_text())
        self.assertEqual(meta['detailed_arm_default'],'r')
    def test_injected_bit51_change_is_localized_and_independently_observed(self):
        original=self.heads['r'].joint_head.forward
        def corrupt(*args,**kwargs):
            value=original(*args,**kwargs)
            self.example.common['ids'].flatten()[0].bitwise_xor_(1<<51)
            return value
        self.heads['r'].joint_head.forward=corrupt
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'Frozen routing'):
            with self.w:self.infer()
        window=json.loads((self.root/'operator_window.json').read_text())
        self.assertEqual(window['previous']['position']['watch_stage'],'r.joint_head_before')
        self.assertEqual(window['current']['position']['watch_stage'],'r.joint_head_after')
        r=self.report()
        self.assertFalse(r['device_equal_before_cpu_snapshot'])
        self.assertFalse(r['device_equal_after_detection']); self.assertTrue(r['mirror_cpu_equals_initial'])
        self.assertEqual([x['changed_elements'] for x in r['reads']],[1,1])
        s=torch.load(self.root/'parent_ids_corroboration_cpu.pt',weights_only=True)
        self.assertEqual(int(s['reference_cpu'].flatten()[0]) ^ int(s['repeat_cpu_0'].flatten()[0]),1<<51)
    def test_copy_only_injected_corruption_preserves_disagreement_in_evidence(self):
        original=torch.Tensor.to; ptr=self.example.common['ids'].data_ptr(); injected=[]
        def copy(self_tensor,*args,**kwargs):
            value=original(self_tensor,*args,**kwargs)
            if (not injected and self_tensor.data_ptr()==ptr and kwargs.get('copy') is True
                    and self.position.get('watch_stage')=='r.joint_head_after'):
                value.flatten()[0].bitwise_xor_(1<<51); injected.append(True)
            return value
        with torch.no_grad(),mock.patch.object(torch.Tensor,'to',copy),self.assertRaisesRegex(ValueError,'Frozen routing'):
            with self.w:self.infer()
        r=self.report(); self.assertTrue(injected)
        self.assertTrue(r['device_equal_before_cpu_snapshot']); self.assertTrue(r['device_equal_after_detection'])
        self.assertTrue(all(x['equals_initial'] for x in r['reads']))
        self.assertTrue(r['mirror_cpu_equals_initial']); self.assertIn('unresolved',r['causal_conclusion'])
    def test_changed_mirror_is_not_mislabeled_as_changed_parent(self):
        with torch.no_grad(), self.assertRaisesRegex(ValueError,'mirror comparison'):
            with self.w:
                self.runtime.prepare(None)
                self.w.reference_gpu.flatten()[0].bitwise_xor_(1<<51)
                self.w.snapshot('forward_before')
        r=self.report(); self.assertFalse(r['mirror_cpu_equals_initial'])
        self.assertTrue(all(x['equals_initial'] for x in r['reads']))
    def test_cuda_like_exception_does_not_attempt_corroboration(self):
        error=RuntimeError('CUDA error: synthetic device-side assert')
        self.heads['r']._read=lambda *a,**kw:(_ for _ in ()).throw(error)
        with torch.no_grad(),self.assertRaises(RuntimeError) as caught:
            with self.w:self.infer()
        self.assertIs(caught.exception,error); self.assertFalse(self.w.confirmed)
        self.assertFalse((self.root/'parent_ids_corroboration_cpu.pt').exists())


class PrefixReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f=fixtures.EvaluationContracts(); self.f.setUp(); self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime=self.f.runtime(); self.names=['test%04d'%i for i in range(contexts.COUNT-1)]+[contexts.TARGET]
        self.root=self.f.folder.parent; self.folder=self.root/'test_step_001500'; self.state=self.f.legacy.state
    def infer(self):
        observer=EvaluationObserver(self.root,dict(index=-1,image=None,active_arm=None,stage='loading'))
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No updates')), \
                track_evaluation(self.runtime,self.f.factory,evaluator,self.names,self.state,observer):
            return scoring.evaluate_test(self.runtime,self.f.heads,self.names,self.folder,seed=3407,
                original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,evaluator=evaluator,
                batches=lambda:self.runtime.batches(self.names,train=False,seed=3407),step=1500)
    def finalize(self, r):
        return reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
            checks={k:True for k in REQUIRED_CHECKS},provenance={'checkpoint_step':1500},full=False,evidence=None)
    def write_meta(self):
        write_json(self.root/'manifest.json',dict(names=self.names,canonical_test_names=self.names+['extra']))
        write_json(self.root/'operator_observer_summary.json',dict(boundary_calls=1,bindings_restored=True,detailed_arm_default='r'))
        write_json(self.root/'corroboration_observer_summary.json',dict(mirror_images=contexts.COUNT,
            device_comparisons=contexts.COUNT,corroboration_triggered=False,evidence_error=None,bindings_restored=True))
    def test_complete356_stays_diagnostic_not_full_test(self):
        rng=self.state.state_fingerprint(self.state.capture_rng()); r=self.infer(); self.write_meta()
        final=self.finalize(r)
        self.assertTrue(final['diagnostic_complete']); self.assertFalse(final['complete'])
        self.assertFalse(final['full_test_evidence']); self.assertTrue(final['sanitizer_check_pending'])
        self.assertEqual(final['checkpoint_selection'],contexts.SELECTION)
        self.assertEqual(rng,self.state.state_fingerprint(self.state.capture_rng()))
        self.names[-1]='wrong_target'
        with self.assertRaisesRegex(ValueError,'identity'): self.finalize(final)
    def test_truncated_journal_is_rejected(self):
        r=self.infer(); self.write_meta()
        p=self.folder/'images.jsonl'; lines=p.read_text().splitlines(); p.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'journal changed'): self.finalize(r)


class Incident(unittest.TestCase):
    def test_incident_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); names=['test%04d'%i for i in range(878)]; names[contexts.INDEX]=contexts.TARGET
            pos=dict(index=contexts.INDEX,image=contexts.TARGET,active_arm='r')
            write_json(p/'summary.json',dict(checkpoint_step=1500,phase='failed',completed_images=contexts.INDEX,
                first_failure=dict(position=pos,error='Frozen routing tensors changed: parent.ids')))
            metadata=dict(dtype='torch.int64',shape=[9,65536],version=0,storage_ptr=1)
            w={k:dict(position=dict(pos,watch_stage=stage),groups=[dict(keys=['parent.ids'],metadata=metadata,
                statistics=dict(sha256=k))]) for k,stage in [('previous','forward_before'),('current','forward_after')]}
            write_json(p/'operator_window.json',w); contexts.check_incident(p,names)
            w['current']['position']['active_arm']='d'; write_json(p/'operator_window.json',w)
            with self.assertRaisesRegex(ValueError,'window'):contexts.check_incident(p,names)


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
    def test_only_zero_summary_is_clean(self):
        p=self.root/'sanitizer.log'
        for data,want in [('no CUDA work',False),('========= ERROR SUMMARY: 0 errors',True),
                          ('ERROR SUMMARY: 1 error\nERROR SUMMARY: 0 errors',False)]:
            p.write_text(data); self.assertEqual(entry.sanitizer_summary(p)['zero_errors_reported'],want)
        self.assertFalse(entry.sanitizer_summary(self.root/'missing')['zero_errors_reported'])
    def test_command_is_fixed_single_prefix_under_memcheck(self):
        c=entry.command_for(self.root,{'sanitizer':{'path':'/usr/local/cuda/bin/compute-sanitizer'}})
        self.assertEqual(c[0],'/usr/local/cuda/bin/compute-sanitizer')
        self.assertEqual(c[c.index('--tool')+1],'memcheck'); self.assertIn('86',c)
        self.assertEqual(c[-3:],['case','--root',str(self.root)])
    def test_real_child_timeout_kills_and_preserves_log(self):
        code='import time; print("started",flush=True); time.sleep(30)'
        with mock.patch.object(entry,'command_for',return_value=[sys.executable,'-c',code]),contextlib.redirect_stdout(io.StringIO()):
            r=entry.supervise(self.root,dict(timeout_seconds=.05),{'requested':False})
        self.assertEqual(r['supervisor_reason'],'time_budget_exceeded'); self.assertNotEqual(r['returncode'],0)
        self.assertTrue((self.root/contexts.CASE/'case.log').is_file())
    def test_timeout_cannot_promote_stale_success_and_no_retry(self):
        request=dict(evidence_files={}); record=dict(returncode=0,supervisor_reason='time_budget_exceeded',sanitizer={'zero_errors_reported':True})
        with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
                mock.patch.object(entry,'read_request',return_value=request),mock.patch.object(entry,'supervise',return_value=record) as run, \
                mock.patch.object(entry,'accept') as accept,contextlib.redirect_stdout(io.StringIO()):
            result=entry.coordinate(self.root)
        self.assertEqual(result,2); self.assertEqual(run.call_count,1); accept.assert_not_called()
        s=json.loads((self.root/'summary.json').read_text())
        self.assertTrue(s['job_finished']); self.assertFalse(s['diagnostic_complete']); self.assertFalse(s['full_test_evidence'])
    def test_python_success_without_sanitizer_summary_is_not_pass(self):
        with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
                mock.patch.object(entry,'read_request',return_value=dict(evidence_files={})), \
                mock.patch.object(entry,'supervise',return_value=dict(returncode=0,supervisor_reason=None,sanitizer={'zero_errors_reported':False})), \
                mock.patch.object(entry,'accept') as accept,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.coordinate(self.root),2)
        accept.assert_not_called()


if __name__=='__main__': unittest.main()
