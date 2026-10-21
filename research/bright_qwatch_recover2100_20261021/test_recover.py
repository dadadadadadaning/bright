"""Packaged CPU tests: restoration, real six-arm updates, fault evidence and gates.
Synthetic runtime data; no claim of reproducing the remote CUDA failure.
"""
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
ARCHIVE=HERE/'bright_qwatch_recover2100_v1.pyz'
sys.path.insert(0,str(ARCHIVE))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'],os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import torch
import recover_entry as entry
import recover_training as training
import parent_q_observer as watch
from passive_routes import PassiveRoutes
from evaluate_checkpoint import ARMS,write_json
from grid_probe import sha256
import test_resume_integration as fixture
import test_supports as real
import ablation_runner
from supports import SupportFactory


def counters():
    return dict(paired_updates_observed=0,snapshots=0,content_changes=0,failures=0,evidence_error=None)


class RealModel(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.runtime,self.example=real.make_fixture()
        self.runtime.loss=lambda prob,target: fixture.pixel_nll(prob,target)+fixture.auxiliary_loss(prob,target)
        self.runtime.deps=SimpleNamespace(region_nll=fixture.pixel_nll)
        self.factory=SupportFactory(self.runtime,seed=3407)
        self.heads={a:real._model() for a in ARMS}
        self.opts={a:torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=.005) for a,h in self.heads.items()}
        self.state=fixture.CPUState()
        self.count=counters()
        self.journal=io.StringIO()
        self.monitor=watch.ParentQObserver(torch,self.example,self.heads,self.opts,self.factory,self.runtime,
                                          self.root,1,self.journal,self.count)
    def update(self):
        return ablation_runner.update_arms(self.heads,self.opts,self.example,self.runtime,self.factory,
                                          self.state,step=1,multiplier=.2)
    def test_real_model_six_arm_updates_are_bit_exact_with_observer_and_passive_routes(self):
        rng=self.state.capture_rng()
        initial={a:copy.deepcopy(h.state_dict()) for a,h in self.heads.items()}
        baseline=self.update()
        after={a:self.state.state_fingerprint((h.state_dict(),self.opts[a].state_dict())) for a,h in self.heads.items()}
        rng_after=self.state.state_fingerprint(self.state.capture_rng())
        for a,h in self.heads.items():
            h.load_state_dict(initial[a]); self.opts[a]=torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=.005)
        self.state.restore_rng(rng)
        before_q=self.example.common['q'].clone()
        with PassiveRoutes(torch,real.original_model.JointRegionMemoryClassifier,self.root,self.monitor.position),self.monitor:
            observed=self.update()
        self.assertEqual(observed,baseline)
        for a,h in self.heads.items():
            self.assertEqual(self.state.state_fingerprint((h.state_dict(),self.opts[a].state_dict())),after[a])
        self.assertEqual(self.state.state_fingerprint(self.state.capture_rng()),rng_after)
        self.assertTrue(torch.equal(self.example.common['q'],before_q))
        self.assertEqual(self.count['content_changes'],0)
        stages=[json.loads(s)['stage'] for s in self.journal.getvalue().splitlines()]
        for s in ('backward_begin','optimizer_before_after_backward','optimizer_after'):
            self.assertEqual(stages.count(s),6)
        self.assertNotIn('forward',vars(self.factory)); self.assertNotIn('prepare',vars(self.factory))
        self.assertEqual(json.loads((self.root/'route_observer_summary.json').read_text())['failures'],0)
    def test_mutation_during_e_optimizer_is_captured_before_f(self):
        q_before=self.example.common['q'].clone()
        handle=self.opts['e'].register_step_post_hook(lambda *args:self.example.common['q'].add_(.01))
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(ValueError,'optimizer_after'):
            with self.monitor:self.update()
        report=json.loads((self.root/'parent_q_failure.json').read_text())
        self.assertEqual(report['position']['active_arm'],'e')
        self.assertEqual(report['position']['stage'],'optimizer_after')
        self.assertFalse(self.opts['f'].state)
        saved=torch.load(self.root/'parent_q_change_cpu.pt',weights_only=False)
        self.assertTrue(torch.equal(saved['before']['q'],q_before))
        self.assertTrue(torch.equal(saved['after']['q'],self.example.common['q']))
        self.assertNotIn('forward',vars(self.factory))
        self.assertEqual(len(self.opts['e']._optimizer_step_post_hooks),1)
    def test_primary_cuda_like_error_does_not_trigger_more_tensor_snapshots(self):
        error=RuntimeError('synthetic device-side assert')
        self.factory.forward=mock.Mock(side_effect=error)
        original=self.monitor.snapshot
        stages=[]
        def capture(stage):stages.append(stage); return original(stage)
        self.monitor.snapshot=capture
        with self.assertRaises(RuntimeError) as caught:
            with self.monitor:self.update()
        self.assertIs(caught.exception,error)
        self.assertEqual(stages[-1],'forward_before')
        self.assertEqual(self.count['failures'],1)
        self.assertEqual(json.loads((self.root/'parent_q_failure.json').read_text())['position']['active_arm'],'e')
    def test_original_failed_sum_uses_original_predicate_and_saves_original_operands(self):
        self.example.common['q'].mul_(2)
        q=self.example.common['q'].clone()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'sum to one'):
            with PassiveRoutes(torch,real.original_model.JointRegionMemoryClassifier,self.root,self.monitor.position),self.monitor:
                # Isolate classifier entry from support construction.
                self.heads['b'](**self.example.common,**self.example.supports['b'])
        record=json.loads((self.root/'route_failure.json').read_text())
        self.assertFalse(record['original_allclose'][0]['result'])
        self.assertFalse(record['original_allclose'][0]['cpu_check_of_original_operands'])
        self.assertTrue(torch.equal(q,self.example.common['q']))
    def test_nonfinite_metadata_is_serializable_without_changing_input(self):
        self.example.common['q'][0,0,0]=float('nan')
        self.monitor.snapshot('before'); self.monitor.snapshot('unchanged')
        self.assertEqual(self.count['content_changes'],0)
        self.assertEqual(self.monitor.latest['q_nonfinite_count'],1)
        self.assertIsNone(self.monitor.latest['q_min'])


class RecoveryWorker(unittest.TestCase):
    def run_worker(self,fail=False,mutation=False):
        f=fixture.ResumeIntegrationTests(); f.setUp(); self.addCleanup(f.doCleanups)
        c=f.context(fail=fail); bridge=c['bridge']; named,loader=bridge.named,bridge.load_runtime
        root=f.base/'recover'; root.mkdir()
        original_prepare=fixture.TinyRuntime.prepare
        original_forward=fixture.TinyFactory.forward
        def prepare(runtime,batch):
            e=original_prepare(runtime,batch)
            e.common=dict(q=torch.full((1,2,4),.5),ids=torch.zeros(1,2,4,dtype=torch.long))
            return e
        def forward(factory,head,example,arm,support):
            out=original_forward(factory,head,example,arm,support)
            if mutation and example.image==f.order[1002] and arm=='e':example.common['q'].mul_(1.1)
            return out
        with mock.patch.dict(sys.modules),mock.patch.dict(os.environ),contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(torch.cuda,'is_available',return_value=True), \
             mock.patch.object(torch.cuda,'is_current_stream_capturing',return_value=False), \
             mock.patch.object(fixture.TinyRuntime,'prepare',prepare), \
             mock.patch.object(fixture.TinyFactory,'forward',forward), \
             mock.patch.object(watch,'PassiveRoutes',return_value=contextlib.nullcontext()):
            # Synthetic head has no routing. Real routes are exercised above.
            with watch.observe_bindings(c,root):code=training.run_training(c,root)
        self.assertIs(bridge.named,named); self.assertIs(bridge.load_runtime,loader)
        self.assertEqual(f.snapshot(f.root),f.before)
        return f,root,code
    def test_packaged_worker_preserves_checkpoint_optimizer_rng_and_order(self):
        f,root,code=self.run_worker()
        self.assertEqual(code,0,(root/'traceback.txt').read_text() if (root/'traceback.txt').exists() else '')
        f.assert_checkpoint(root/'checkpoints/checkpoint_001004.pt',1004); f.assert_journal(root,1004)
        self.assertEqual(f.evaluations,[])
        s=json.loads((root/'parent_q_observer_summary.json').read_text())
        self.assertEqual(s['paired_updates_observed'],4); self.assertEqual(s['failures'],0)
        self.assertFalse(json.loads((root/'summary.json').read_text())['test_accessed'])
    def test_backward_failure_preserves_last_paired_boundary(self):
        f,root,code=self.run_worker(fail=True)
        self.assertNotEqual(code,0)
        s=json.loads((root/'summary.json').read_text())
        self.assertEqual(s['last_complete_step'],1002); self.assertIn('synthetic B backward failure',s['error'])
        f.assert_checkpoint(root/'checkpoints/checkpoint_001002.pt',1002); f.assert_journal(root,1002)
        p=json.loads((root/'parent_q_failure.json').read_text())
        self.assertEqual(p['position']['active_arm'],'b'); self.assertEqual(p['position']['stage'],'backward_begin')
    def test_q_change_rejects_partial_update_without_checkpoint_or_journal_promotion(self):
        f,root,code=self.run_worker(mutation=True)
        self.assertNotEqual(code,0)
        s=json.loads((root/'summary.json').read_text()); self.assertEqual(s['last_complete_step'],1002)
        self.assertIn('Frozen parent Q/IDs changed',s['error'])
        f.assert_checkpoint(root/'checkpoints/checkpoint_001002.pt',1002); f.assert_journal(root,1002)
        self.assertFalse((root/'checkpoints/checkpoint_001003.pt').exists())
        self.assertEqual(f.evaluations,[])


class Gates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
    def good_observations(self):
        write_json(self.root/'parent_q_observer_summary.json',dict(paired_updates_observed=400,snapshots=20000,
            content_changes=0,failures=0,evidence_error=None,route_values_modified=False))
        write_json(self.root/'route_observer_summary.json',dict(calls=10000,failures=0,evidence_error=None,bindings_restored=True))
        return dict(last_complete_step=2500,training_complete=True,new_updates_per_arm={a:400 for a in ARMS})
    def test_exact400_and_successful_observations_required(self):
        s=self.good_observations(); entry.validate_observations(self.root,s)
        s['new_updates_per_arm']['e']=401
        with self.assertRaisesRegex(ValueError,'400'):entry.validate_observations(self.root,s)
        s=self.good_observations(); p=self.root/'parent_q_observer_summary.json'; d=json.loads(p.read_text())
        d['content_changes']=1; write_json(p,d)
        with self.assertRaisesRegex(ValueError,'observation'):entry.validate_observations(self.root,s)
    def test_training_error_prevents_any_evaluation_or_test(self):
        calls=[]
        def run(command,**kwargs):
            calls.append(command[-1]); write_json(self.root/'summary.json',dict(phase='failed',error='route error'))
            return SimpleNamespace(returncode=1)
        with mock.patch.object(entry,'leases',return_value=contextlib.nullcontext()), \
             mock.patch.object(entry,'read_request',return_value={}),mock.patch.object(entry.subprocess,'run',side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):self.assertNotEqual(entry.coordinate(self.root),0)
        self.assertEqual(calls,['train']); self.assertFalse(json.loads((self.root/'workflow_summary.json').read_text())['test_complete'])
    def test_eval_error_after400_stops_before_test_without_retraining(self):
        calls=[]
        def run(command,**kwargs):
            stage=command[-1]; calls.append(stage)
            if stage=='train':
                s=self.good_observations(); s['phase']='training_complete_evaluation_pending'; write_json(self.root/'summary.json',s)
                return SimpleNamespace(returncode=0)
            out=self.root/'assessments'/stage; out.mkdir(parents=True)
            write_json(out/'summary.json',dict(complete=False,phase='failed',error='synthetic evaluation failure'))
            return SimpleNamespace(returncode=2)
        with mock.patch.object(entry,'leases',return_value=contextlib.nullcontext()), \
             mock.patch.object(entry,'read_request',return_value={}),mock.patch.object(entry.subprocess,'run',side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):self.assertNotEqual(entry.coordinate(self.root),0)
        self.assertEqual(calls,['train','fixed2000'])
        s=json.loads((self.root/'workflow_summary.json').read_text())
        self.assertTrue(s['training_complete']); self.assertFalse(s['test_complete'])
    def test_fixed2000_uses_certified2000_not_restore2100(self):
        c=object()
        import assess_endpoint
        with mock.patch.object(entry,'read_request'),mock.patch.object(entry,'certified_boundary',return_value=c), \
             mock.patch.object(entry,'initial_context',side_effect=AssertionError('2100 is not Val2000')), \
             mock.patch.object(entry,'inactive_parent',return_value=contextlib.nullcontext()), \
             mock.patch.object(assess_endpoint,'run_assessment',return_value=0) as call:
            self.assertEqual(entry.assess(self.root,'fixed2000'),0)
        self.assertIs(call.call_args.args[0],c)
    def test_archive_preserves_every_inherited_byte_and_matches_sources(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            ledger=json.loads(z.read('QWATCH_BASE.json')); self.assertEqual(ledger['source_sha256'],entry.PARENT_ARCHIVE_SHA)
            for n,h in ledger['files'].items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h,n)
            for n in ('recover_entry.py','recover_training.py','parent_q_observer.py'):
                self.assertEqual(z.read(n),(HERE/n).read_bytes())


class ParentGate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        self.cp=self.root/'checkpoints/checkpoint_002100.pt'; self.cp.parent.mkdir(); self.cp.write_bytes(b'synthetic fixture')
        self.c=dict(parent=dict(path=self.cp,step=2100,sha256=entry.PARENT_CP_SHA),plan=dict(endpoint_step=2500),
            lineage=dict(seed=3407),evidence_files={})
        self.boundary=dict(lineage=dict(seed=3407),evidence_files={},boundary_certificate=dict(step=2000))
        self.s=dict(phase='failed',last_complete_step=2135,failure_phase='update_2136',optimizer_steps_returned_in_observed_step=['e'],
            last_marker=dict(step=2136,arm='f',image='la_palma-volcano_00000900',phase='forward_begin',completed_arms=['e']))
        write_json(self.root/'summary.json',self.s)
        write_json(self.cp.with_suffix('.pt.json'),dict(payload_fingerprint=entry.PARENT_PAYLOAD_SHA,
            roundtrip_verified=True,updates_per_arm={a:2100 for a in ARMS}))
        write_json(self.root/'workflow_summary.json',dict(phase='failed',failed_stage='train',test_complete=False))
        for n in ('markers.jsonl','traceback.txt','request.json','trainer.pyz'):(self.root/n).write_text('synthetic')
    def run_context(self):
        def digest(p):return entry.PARENT_ARCHIVE_SHA if Path(p).name=='trainer.pyz' else sha256(p)
        with mock.patch.object(entry,'RECOVERY_PARENT',self.root),mock.patch.object(entry,'certified_boundary',return_value=self.boundary), \
             mock.patch.object(entry,'prepare',return_value=copy.deepcopy(self.c)) as prepare,mock.patch.object(entry,'sha256',side_effect=digest):
            c=entry.initial_context()
        self.assertEqual(prepare.call_args.args[0].checkpoint_every,10)
        return c
    def test_accepts_only_recorded_paired2100_and_preserves2000_certificate(self):
        c=self.run_context(); self.assertEqual(c['evaluation_boundary_step'],2000)
        self.assertEqual(c['parent']['step'],2100); self.assertFalse(c['parent_q_observation']['historical_failure_resolved'])
    def test_rejects_unpinned_checkpoint(self):
        self.c['parent']['sha256']='different'
        with self.assertRaisesRegex(ValueError,'pinned'):self.run_context()
    def test_rejects_promoted_partial2136(self):
        self.c['parent']['step']=2136
        with self.assertRaisesRegex(ValueError,'pinned'):self.run_context()
    def test_rejects_different_failure(self):
        self.s['last_marker']['arm']='d'; write_json(self.root/'summary.json',self.s)
        with self.assertRaisesRegex(ValueError,'pinned'):self.run_context()


if __name__=='__main__':unittest.main()
