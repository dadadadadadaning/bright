"""CPU evidence/overnight supervisor tests, not a remote CUDA reproduction."""
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

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'bright_test_route_night_v1.pyz'
sys.path.insert(0,str(ARCHIVE))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'],os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import torch
import night_entry as entry
import night_assessment as assessment
from night_routes import TestRouteWatch,describe
from night_config import CASES,TARGET,select_case,SOURCE_SHA
from passive_routes import PassiveRoutes
from evaluate_checkpoint import ARMS,REQUIRED_CHECKS,write_json
from evaluation_observer import EvaluationObserver,track_evaluation
import test_endpoint as scoring
import evaluation as evaluator
import test_supports as real
from supports import SupportFactory
import test_resume_integration as fixture_train
import test_evaluation as fixture_eval


def canonical():
    names=['test%04d'%i for i in range(878)];names[621]=TARGET
    return names


class RealRouteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.runtime,self.example=real.make_fixture();self.example.image=TARGET
        self.runtime.prepare=lambda batch:self.example
        self.factory=SupportFactory(self.runtime,3407)
        self.heads={a:real._model().eval() for a in ARMS}
        self.evaluator=SimpleNamespace(_validate_probability=evaluator._validate_probability)
        self.position=dict(index=-1,image=TARGET,active_arm=None,stage='loading')
    def watch(self,mode):
        return TestRouteWatch(torch,self.runtime,self.factory,self.evaluator,self.root,self.position,[TARGET],canonical(),mode,TARGET)
    def run_forward(self):
        e=self.runtime.prepare(TARGET);support=self.factory.prepare(e,ARMS);result={}
        for a in ARMS:
            p=self.factory.forward(self.heads[a],e,a,support[a])['probabilities']
            self.evaluator._validate_probability(p,e.target,a);result[a]=p.detach().clone()
        return result
    def test_original_six_arm_predictions_unchanged_and_real_aliases_observed(self):
        state=fixture_train.CPUState();before=state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()})
        with torch.no_grad():expected=self.run_forward()
        with torch.no_grad(),PassiveRoutes(torch,real.original_model.JointRegionMemoryClassifier,self.root,self.position),self.watch('watched'):
            got=self.run_forward()
        for a in ARMS:self.assertTrue(torch.equal(got[a],expected[a]))
        self.assertEqual(before,state.state_fingerprint({a:h.state_dict() for a,h in self.heads.items()}))
        rows=[json.loads(s) for s in (self.root/'route_snapshots.jsonl').read_text().splitlines()]
        row=next(r for r in rows if r['position']['watch_stage']=='supports_after')
        group=next(g for g in row['groups'] if 'arm.b.route_q' in g['keys'])
        self.assertIn('arm.d.route_q',group['keys']);self.assertIn('arm.r.route_q',group['keys'])
        self.assertEqual(row['position']['canonical_index'],621)
        self.assertTrue((self.root/'target_routes_cpu.pt').is_file())
        self.assertEqual(assessment.validate_route_evidence(self.root,[TARGET],'watched')['images_completed'],1)
    def corrupt_after_b(self):
        original=self.factory.forward
        def forward(head,e,a,s):
            value=original(head,e,a,s)
            if a=='b':e.supports['b']['route_q'].mul_(2)
            return value
        self.factory.forward=forward
    def test_b_changes_shared_child_tensor_detected_before_d(self):
        self.corrupt_after_b();calls=[];original=self.factory.forward
        def call(head,e,a,s):calls.append(a);return original(head,e,a,s)
        self.factory.forward=call
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'Frozen routing tensors changed'):
            with self.watch('watched'):self.run_forward()
        self.assertEqual(calls,['e','f','b'])
        d=json.loads((self.root/'route_change.json').read_text())
        self.assertIn('arm.d.route_q',d['changed_keys']);self.assertEqual(d['position']['active_arm'],'b')
        saved=torch.load(self.root/'route_change_cpu.pt',weights_only=False)
        self.assertTrue(torch.equal(saved['after']['arm.d.route_q'],saved['before']['arm.d.route_q']*2))
    def test_passive_original_d_check_captures_actual_bad_child_operands(self):
        self.corrupt_after_b()
        with contextlib.redirect_stdout(io.StringIO()),torch.no_grad(),self.assertRaisesRegex(ValueError,'sum to one'):
            with PassiveRoutes(torch,real.original_model.JointRegionMemoryClassifier,self.root,self.position),self.watch('passive'):
                self.run_forward()
        d=json.loads((self.root/'route_failure.json').read_text())
        self.assertEqual(d['position']['active_arm'],'d');self.assertFalse(d['original_allclose'][0]['result'])
        data=torch.load(self.root/'route_failure_cpu.pt',weights_only=False)
        self.assertEqual(data['tensors']['q'].shape[1],18)
        self.assertFalse(d['original_allclose'][0]['cpu_check_of_original_operands'])
        self.assertEqual(json.loads((self.root/'route_watch_summary.json').read_text())['snapshots'],0)
    def test_cuda_like_exception_keeps_primary_and_does_not_copy_after_failure(self):
        original=self.factory.forward;error=RuntimeError('synthetic device-side assert')
        def forward(head,e,a,s):
            if a=='d':raise error
            return original(head,e,a,s)
        self.factory.forward=forward;watch=self.watch('watched');stages=[];snapshot=watch.snapshot
        def capture(stage):stages.append(stage);return snapshot(stage)
        watch.snapshot=capture
        with torch.no_grad(),self.assertRaises(RuntimeError) as caught:
            with watch:self.run_forward()
        self.assertIs(caught.exception,error);self.assertEqual(stages[-1],'forward_before')
        self.assertFalse(json.loads((self.root/'watch_failure.json').read_text())['new_tensor_reads_after_failure'])
        self.assertNotIn('prepare',vars(self.factory))
    def test_nonfinite_statistics_remain_json_serializable(self):
        x=torch.tensor([[[float('nan')],[float('inf')]]])
        stats=describe(torch,x);self.assertEqual(stats['nonfinite_count'],2)
        json.dumps(stats,allow_nan=False)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture_eval.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f=fixture_eval.EvaluationContracts();self.f.setUp();self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime=self.f.runtime();self.factory=self.f.factory;self.root=self.f.folder.parent
        prepare=self.runtime.prepare;make=self.factory.prepare;forward=self.factory.forward
        def extract(name):
            e=prepare(name);q=torch.full((1,9,4),1/9);ids=torch.zeros_like(q,dtype=torch.long)
            e.common.update(q=q,ids=ids)
            child=torch.cat((q/2,q/2),1);child_ids=torch.cat((ids,ids+1),1)
            for a in ('b','d'):e.supports[a].update(route_q=child,route_ids=child_ids)
            return e
        def supports(e,arms):
            result=make(e,arms);q=e.common['q'];ids=e.common['ids']
            single=torch.cat((q,torch.zeros_like(q)),1);single_ids=torch.cat((ids,ids+1),1)
            for a,s in result.items():s.update(route_q=single if a in ('e','f') else e.supports['b']['route_q'],route_ids=single_ids)
            return result
        def model(head,e,a,s):return forward(head,e,a,dict(shift=s['shift']))
        self.runtime.prepare=extract;self.factory.prepare=supports;self.factory.forward=model
    def infer(self,root,names,watched):
        root.mkdir();folder=root/'test_step_002500';position=dict(index=-1,image=None,active_arm=None,stage='loading')
        observer=EvaluationObserver(root,position)
        watch=TestRouteWatch(torch,self.runtime,self.factory,evaluator,root,position,names,canonical(),'watched',TARGET)
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Test optimizer called')), \
             track_evaluation(self.runtime,self.factory,evaluator,names,self.f.legacy.state,observer),watch if watched else contextlib.nullcontext():
            result=scoring.evaluate_test(self.runtime,self.f.heads,names,folder,seed=3407,original_train=self.f.original,
                legacy=self.f.legacy,factory=self.factory,evaluator=evaluator,
                batches=lambda:self.runtime.batches(names,train=False,seed=3407))
        return result
    def test_all878_metrics_transitions_and_rng_equal_without_updates(self):
        names=canonical();state=self.f.legacy.state
        before=state.state_fingerprint(state.capture_rng());heads=state.state_fingerprint({a:h.state_dict() for a,h in self.f.heads.items()})
        expected=self.infer(self.root/'plain',names,False)
        got=self.infer(self.root/'watched',names,True)
        self.assertEqual(got['metrics'],expected['metrics']);self.assertEqual(got['transitions'],expected['transitions'])
        self.assertEqual(got['image_count'],878);self.assertEqual(self.runtime.reference_calls,0)
        self.assertEqual(before,state.state_fingerprint(state.capture_rng()))
        self.assertEqual(heads,state.state_fingerprint({a:h.state_dict() for a,h in self.f.heads.items()}))
        self.assertEqual(json.loads((self.root/'watched/route_watch_summary.json').read_text())['images_completed'],878)
        root=self.root/'watched';protocol={'synthetic':True}
        write_json(root/'manifest.json',dict(names=names,protocol_info=protocol))
        final=assessment.finish_report(root/'test_step_002500/result.json',got,output=root,names=names,
            checks={k:True for k in REQUIRED_CHECKS},provenance=dict(checkpoint_step=2500,protocol_info=protocol),
            full=True,evidence=None)
        self.assertTrue(final['complete']);self.assertTrue(final['full_test_evidence'])
        self.assertTrue((root/'test_step_002500/result.pending.json').is_file())
    def test_prefix_cannot_be_promoted_to_full_test_evidence(self):
        names=canonical()[:622];root=self.root/'prefix';report=self.infer(root,names,True)
        protocol={'synthetic':True};write_json(root/'manifest.json',dict(names=names,protocol_info=protocol))
        path=root/'test_step_002500/result.json';checks={k:True for k in REQUIRED_CHECKS}
        p=dict(checkpoint_step=2500,protocol_info=protocol)
        with self.assertRaisesRegex(ValueError,'all878'):
            assessment.finish_report(path,report,output=root,names=names,checks=checks,provenance=p,full=True,evidence=None)
        final=assessment.finish_report(path,report,output=root,names=names,checks=checks,provenance=p,full=False,evidence=None)
        self.assertFalse(final['complete']);self.assertTrue(final['diagnostic_complete']);self.assertFalse(final['full_test_evidence'])


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.request=dict(evidence_files={},case_timeout_seconds=7200,total_timeout_seconds=28800)
    def coordinate(self,outcomes):
        calls=[]
        def run(root,case,budget,stop):calls.append(case);return dict(outcomes.get(case,dict(returncode=0,diagnostic_complete=True,supervisor_reason=None)))
        with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()),mock.patch.object(entry,'read_request',return_value=self.request), \
             mock.patch.object(entry,'supervise',side_effect=run),mock.patch.object(entry,'validate_case_success'),contextlib.redirect_stdout(io.StringIO()):
            code=entry.coordinate(self.root)
        return code,calls,read_json(self.root/'summary.json')
    def test_passive_failure_is_retained_and_independent_watched_case_can_continue(self):
        code,calls,s=self.coordinate({'prefix_passive':dict(returncode=2,diagnostic_complete=False,supervisor_reason=None)})
        self.assertEqual(code,0);self.assertEqual(calls,list(CASES));self.assertTrue(s['test_complete'])
        self.assertEqual(s['diagnostic_failures'],['prefix_passive']);self.assertFalse(s['historical_failure_resolved'])
    def test_watched_failure_blocks_full_test(self):
        code,calls,s=self.coordinate({'prefix_watched':dict(returncode=2,diagnostic_complete=False,supervisor_reason=None)})
        self.assertNotEqual(code,0);self.assertEqual(calls,list(CASES[:3]));self.assertFalse(s['test_complete'])
    def test_timeout_cannot_promote_stale_success_or_launch_full_test(self):
        code,calls,s=self.coordinate({'prefix_watched':dict(returncode=-15,diagnostic_complete=True,supervisor_reason='time_budget_exceeded')})
        self.assertEqual(calls,list(CASES[:3]));self.assertFalse(s['test_complete'])
    def test_setup_integrity_failure_aborts_remaining_cases(self):
        code,calls,s=self.coordinate({'target_watched':dict(returncode=3,diagnostic_complete=False,supervisor_reason=None)})
        self.assertEqual(calls,['target_watched']);self.assertTrue(s['night_job_finished']);self.assertFalse(s['complete'])
    def test_real_child_timeout_is_bounded_and_records_partial_log(self):
        (self.root/'runner.pyz').write_text('import time\nprint("synthetic child started",flush=True)\ntime.sleep(60)\n')
        with contextlib.redirect_stdout(io.StringIO()):r=entry.supervise(self.root,'target_watched',.05,dict(requested=False))
        self.assertEqual(r['supervisor_reason'],'time_budget_exceeded');self.assertNotEqual(r['returncode'],0)
        self.assertIn('synthetic child started',(self.root/'target_watched/case.log').read_text())
    def test_completed_case_requires_all_integrity_and_watch_evidence(self):
        root=self.root;source=root/'source';out=source/'assessments/test2500';out.mkdir(parents=True)
        write_json(out/'manifest.json',dict(names=canonical()))
        case='prefix_watched';folder=root/case;folder.mkdir()
        status=dict(diagnostic_complete=True,image_count=622,complete=False,full_test_evidence=False,optimizer_updates=0,
            integrity_checks={k:True for k in REQUIRED_CHECKS})
        write_json(folder/'summary.json',status)
        write_json(folder/'route_observer_summary.json',dict(calls=10,failures=0,evidence_error=None,bindings_restored=True))
        write_json(folder/'route_watch_summary.json',dict(mode='watched',images_completed=622,content_changes=0,failure=False,
            snapshots=1000,evidence_error=None,bindings_restored=True))
        with mock.patch.object(entry,'SOURCE',source):
            entry.validate_case_success(root,case,dict(returncode=0))
            status['integrity_checks']['rng_restored']=False;write_json(folder/'summary.json',status)
            with self.assertRaisesRegex(ValueError,'integrity'):entry.validate_case_success(root,case,dict(returncode=0))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.names=canonical()
        self.context=dict(parent=dict(sha256='cp2500'),evidence_files={})
        self.failed=self.root/'assessments/test2500';self.failed.mkdir(parents=True)
        write_json(self.root/'workflow_summary.json',dict(training_complete=True,validation_complete=True,test_complete=False,
            failed_stage='test2500',last_complete_step=2500))
        self.summary=dict(phase='failed',completed_images=621,first_failure=dict(position=dict(index=621,image=TARGET,active_arm='d')))
        write_json(self.failed/'summary.json',self.summary)
        write_json(self.failed/'manifest.json',dict(names=self.names,provenance=dict(checkpoint_sha256='cp2500',trainer_sha256=SOURCE_SHA)))
        journal=self.failed/'test_step_002500/images.jsonl';journal.parent.mkdir()
        journal.write_text(''.join(json.dumps(dict(index=i,image=n,confusion={a:[] for a in ('grid_probq',*ARMS)}))+'\n' for i,n in enumerate(self.names[:621])))
        for p in (self.root/'trainer.pyz',self.root/'request.json',self.failed/'evaluation_failure.json',self.failed/'traceback.txt'):p.write_text('fixture')
        for stage,folder in (('val2500','val_step_002500'),('fixed2500','fixed_train_step_002500')):
            d=self.root/'assessments'/stage;d.mkdir();(d/folder).mkdir()
            for n in ('summary.json','manifest.json'):(d/n).write_text('{}')
            for n in ('result.json','result.pending.json','result.json.sha256','images.jsonl'):(d/folder/n).write_text('{}')
    def invoke(self):
        original=entry.sha256
        def digest(p):return SOURCE_SHA if Path(p).name=='trainer.pyz' else original(p)
        with mock.patch.object(entry,'SOURCE',self.root),mock.patch.object(entry,'sha256',side_effect=digest), \
             mock.patch.object(entry.recover_entry,'read_request'),mock.patch.object(entry.recover_entry,'verify_before_test'), \
             mock.patch.object(entry.recover_entry,'endpoint_context',return_value=copy.deepcopy(self.context)):
            return entry.preflight()
    def test_pins_checkpoint_and_complete_prior_evidence_without_promoting_partial_test(self):
        c=self.invoke();self.assertEqual(c['parent']['sha256'],'cp2500');self.assertEqual(c['canonical_test_names'],self.names)
    def test_wrong_order_or_failure_is_rejected(self):
        self.summary['first_failure']['position']['image']='other';write_json(self.failed/'summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'recorded'):self.invoke()
    def test_incomplete_prefix_rejected(self):
        p=self.failed/'test_step_002500/images.jsonl';p.write_text(p.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'length'):self.invoke()
    def test_wrong_checkpoint_rejected(self):
        self.context['parent']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'recorded'):self.invoke()


class ArchiveTests(unittest.TestCase):
    def test_source_and_inherited_bytes_match(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            record=json.loads(z.read('NIGHT_BASE.json'));self.assertEqual(record['source_sha256'],SOURCE_SHA)
            for n,h in record['files'].items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h,n)
            for n in ('night_entry.py','night_assessment.py','night_config.py','night_routes.py'):self.assertEqual(z.read(n),(HERE/n).read_bytes())
    def test_case_selection_never_omits_target_from_prefix_or_full(self):
        for case,count in zip(CASES,(1,622,622,878)):
            names,mode,full=select_case(case,canonical());self.assertEqual(len(names),count);self.assertIn(TARGET,names)
            self.assertEqual(full,case=='full_test_watched')


from grid_probe import read_json
if __name__=='__main__':unittest.main()
