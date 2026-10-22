"""CPU contracts for measurements, original six-arm outputs and bounded execution."""
import contextlib
import ast
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'bright_memcheck_memory_v1.pyz'))
sys.path.extend([os.environ['BRIGHT_RESUME_TEST_SOURCE'], os.environ['BRIGHT_ABLATION_TEST_SOURCE']])
import torch
import memory_entry as entry
import memory_context as contexts
import memory_scoring as scoring
import memory_reports as reports
from memory_trace import MemoryTrace, MemoryRoutes
from evaluate_checkpoint import ARMS, REQUIRED_CHECKS, write_json
from evaluation_observer import EvaluationObserver, track_evaluation
import test_supports as real
from supports import SupportFactory
import evaluation as evaluator
import test_evaluation as fixtures
import test_resume_integration as training_fixture


class FakeCuda:
    def __init__(self): self.calls = 0; self.initialized = True
    def is_initialized(self): return self.initialized
    def memory_stats(self): self.calls += 1; return {'num_ooms': 0}
    def mem_get_info(self): self.calls += 1; self.initialized = True; return 2000, 4000
    def memory_allocated(self): self.calls += 1; return 1000
    def memory_reserved(self): self.calls += 1; return 1500
    def max_memory_allocated(self): self.calls += 1; return 1200
    def max_memory_reserved(self): self.calls += 1; return 1600


class Measurements(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cuda = FakeCuda()
        self.trace = MemoryTrace(SimpleNamespace(cuda=self.cuda, is_grad_enabled=torch.is_grad_enabled), self.root)
        self.addCleanup(self.trace.close)

    def test_no_gpu_queries_after_any_failure(self):
        self.trace.sample('before')
        n = self.cuda.calls
        self.trace.failure('failed', RuntimeError('CUDA synthetic'))
        self.trace.sample('must_be_skipped')
        self.assertEqual(self.cuda.calls, n)
        self.assertEqual(self.trace.rows, 2)
        self.assertFalse(json.loads((self.root/'memory_latest.json').read_text())['cuda_queried_after_failure'])

    def test_early_sampling_preserves_original_determinism_configuration(self):
        # Execute the actual source function whose guard rejected the server
        # run; emulate CUDA's lazy initialization without requiring a GPU.
        path = Path(os.environ.get('BRIGHT_NUMERIC_TEST_FILE',
            '/workspace/bright/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/numeric.py'))
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='configure_runtime']
        self.assertEqual(len(nodes),1)
        self.cuda.initialized = False
        events = []
        def settings():
            self.cuda.initialized = True
            events.append('runtime_initialized')
            return {'configured': True}
        numeric_torch = SimpleNamespace(cuda=self.cuda,
            use_deterministic_algorithms=lambda *a, **kw: events.append('policy_configured'),
            backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=True, benchmark=True)))
        scope = dict(torch=numeric_torch, os=os, runtime_settings=settings,
                     configure_determinism=lambda: self.fail('Unexpected strict override'))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        # This would initialize CUDA in the faulty release and reject the next call.
        self.trace.sample('accidental_early_sample')
        self.assertEqual(self.cuda.calls,0)
        self.assertFalse(self.cuda.initialized)
        with mock.patch.dict(os.environ,{},clear=False):
            self.assertEqual(scope['configure_runtime'](deterministic=False),{'configured':True})
        self.trace.sample('runtime_ready')
        self.assertEqual(events,['policy_configured','runtime_initialized'])
        self.assertGreater(self.cuda.calls,0)
        rows = [json.loads(s) for s in (self.root/'memory.jsonl').read_text().splitlines()]
        self.assertNotIn('allocated_bytes',rows[0])
        self.assertEqual(rows[1]['allocated_bytes'],1000)

    def test_actual_uninitialized_torch_is_not_initialized_by_telemetry(self):
        # On this CPU test environment any attempted CUDA initialization fails.
        self.assertFalse(torch.cuda.is_initialized())
        with tempfile.TemporaryDirectory() as d:
            trace = MemoryTrace(torch,Path(d))
            try: trace.sample('before_runtime')
            finally: trace.close()
            self.assertFalse(torch.cuda.is_initialized())
            self.assertFalse(json.loads((Path(d)/'memory_latest.json').read_text())['cuda_initialized'])

    def fixture(self):
        runtime, example = real.make_fixture()
        example.image = 'image'
        front = torch.nn.Sequential(torch.nn.Identity())
        runtime.bundle = {'front': front}
        def prepare(batch):
            front(example.common['pixel_features'])
            return example
        runtime.prepare = prepare
        factory = SupportFactory(runtime, 3407)
        heads = {a: real._model().eval() for a in ARMS}
        position = dict(index=-1, image='image', active_arm=None, stage='loading')
        ev = SimpleNamespace(_validate_probability=evaluator._validate_probability)
        w = MemoryRoutes(torch, runtime, factory, ev, self.root, position, ['image'], ['image'],
            'watched', contexts.TARGET, heads=heads, trace=self.trace)
        def infer():
            e = runtime.prepare(None)
            support = factory.prepare(e, ARMS)
            outputs = {}
            for arm in ARMS:
                p = factory.forward(heads[arm], e, arm, support[arm])['probabilities']
                ev._validate_probability(p, e.target, arm)
                outputs[arm] = p.clone()
            return outputs
        return runtime, heads, w, infer

    def test_original_six_arm_outputs_parameters_rng_and_hooks_unchanged(self):
        runtime, heads, w, infer = self.fixture()
        state = training_fixture.CPUState()
        before = state.state_fingerprint({a:h.state_dict() for a,h in heads.items()})
        with torch.no_grad(): expected = infer()
        rng = state.state_fingerprint(state.capture_rng())
        with torch.no_grad(), w: actual = infer()
        self.assertEqual(rng, state.state_fingerprint(state.capture_rng()))
        self.assertEqual(before, state.state_fingerprint({a:h.state_dict() for a,h in heads.items()}))
        for a in ARMS: self.assertTrue(torch.equal(expected[a], actual[a]))
        for m in runtime.bundle['front'].modules():
            self.assertFalse(m._forward_pre_hooks); self.assertFalse(m._forward_hooks)
        rows = [json.loads(s) for s in (self.root/'memory.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['grad_enabled'] is False for r in rows))
        self.assertEqual(sum(r['stage']=='arm_forward_after' for r in rows), 6)
        self.assertEqual([r['position']['active_arm'] for r in rows if r['stage']=='arm_forward_before'], list(ARMS))
        self.assertTrue(any(r['stage']=='front.0_after' for r in rows))

    def test_original_oom_exception_preserved_and_no_queries_afterward(self):
        runtime, heads, w, infer = self.fixture()
        error = torch.cuda.OutOfMemoryError('synthetic OOM')
        count_at_failure = []
        def fail(*a, **kw):
            count_at_failure.append(self.cuda.calls)
            raise error
        runtime.bundle['front'][0].forward = fail
        with torch.no_grad(), self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
            with w: infer()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.cuda.calls, count_at_failure[0])
        self.assertTrue(self.trace.failed)
        self.assertFalse(runtime.bundle['front'][0]._forward_hooks)

    def test_digest_uses_peak_and_actual_stage_without_failure_as_measurement(self):
        self.trace.sample('front_before', dict(index=1))
        self.trace.failure('front_failed', RuntimeError('OOM'))
        r = entry.memory_digest(self.root)
        self.assertEqual(r['rows'], 2)
        self.assertEqual(r['at_max']['stage'], 'front_before')
        self.assertEqual(r['peak_allocated_gib'], 1200 / 2**30)


class PrefixReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f = fixtures.EvaluationContracts(); self.f.setUp(); self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime = self.f.runtime(); self.names = ['test0', contexts.PREFIX_LAST]
        self.root = self.f.folder.parent; self.folder = self.root/'test_step_001500'; self.state=self.f.legacy.state
    def evaluate(self):
        observer = EvaluationObserver(self.root, dict(index=-1,image=None,active_arm=None,stage='loading'))
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No updates')), \
                track_evaluation(self.runtime,self.f.factory,evaluator,self.names,self.state,observer):
            return scoring.evaluate_test(self.runtime,self.f.heads,self.names,self.folder,seed=3407,
                original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,evaluator=evaluator,
                batches=lambda:self.runtime.batches(self.names,train=False,seed=3407),step=1500)
    def finalize(self, r):
        write_json(self.root/'manifest.json',dict(names=self.names,canonical_test_names=self.names+['extra']))
        write_json(self.root/'operator_observer_summary.json',dict(boundary_calls=1,bindings_restored=True,detailed_arm_default='r'))
        write_json(self.root/'corroboration_observer_summary.json',dict(mirror_images=2,device_comparisons=2,
            corroboration_triggered=False,evidence_error=None,bindings_restored=True))
        return reports.finish_report(self.folder/'result.json',r,output=self.root,names=self.names,
            checks={k:True for k in REQUIRED_CHECKS},provenance={'checkpoint_step':1500,'sanitizer_tool':None},full=False,evidence=None)
    def test_two_image_success_is_not_full_test_or_hardware_clearance(self):
        rng=self.state.state_fingerprint(self.state.capture_rng())
        r=self.finalize(self.evaluate())
        self.assertTrue(r['diagnostic_complete']); self.assertFalse(r['complete']); self.assertFalse(r['full_test_evidence'])
        self.assertFalse(r['historical_failure_resolved']); self.assertFalse(r['sanitizer_check_pending'])
        self.assertEqual(rng,self.state.state_fingerprint(self.state.capture_rng()))
        self.assertEqual(r['image_count'],2)
    def test_shortened_journal_not_accepted(self):
        r=self.evaluate(); p=self.folder/'images.jsonl'
        p.write_text(p.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'journal changed'): self.finalize(r)


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
    def test_command_difference_is_instrumentation_only(self):
        r={'sanitizer':{'path':'/sanitizer'}}
        plain=entry.command_for(self.root,'plain',r); check=entry.command_for(self.root,'memcheck',r)
        self.assertEqual(plain[:-1],check[-len(plain):-1]); self.assertEqual(check[0],'/sanitizer')
        self.assertNotIn('--force-blocking-launches',check)
    def test_real_timeout_stops_child(self):
        command=[sys.executable,'-c','import time;print("alive",flush=True);time.sleep(30)']
        with mock.patch.object(entry,'command_for',return_value=command),contextlib.redirect_stdout(io.StringIO()):
            r=entry.supervise(self.root,'plain',{'timeout_seconds':.05},{'requested':False})
        self.assertEqual(r['supervisor_reason'],'time_budget_exceeded'); self.assertNotEqual(r['returncode'],0)
        self.assertIn('alive',(self.root/'plain/case.log').read_text())
    def run_pair(self, accepted):
        record=dict(returncode=0,supervisor_reason=None)
        def decision(root, case, record, request): return dict(record,accepted=accepted[case])
        with mock.patch.object(entry.recover_entry,'leases',return_value=contextlib.nullcontext()), \
                mock.patch.object(entry,'read_request',return_value={'evidence_files':{}}), \
                mock.patch.object(entry,'require_idle'), mock.patch.object(entry,'supervise',return_value=record) as run, \
                mock.patch.object(entry,'accept',side_effect=decision),contextlib.redirect_stdout(io.StringIO()):
            code=entry.coordinate(self.root)
        return code,run.call_count,json.loads((self.root/'summary.json').read_text())
    def test_plain_failure_skips_memcheck_without_retry(self):
        code,n,s=self.run_pair({'plain':False})
        self.assertEqual(code,2); self.assertEqual(n,1); self.assertIn('memcheck_skipped',s)
        self.assertFalse(s['diagnostic_complete'])
    def test_pair_completion_keeps_reliability_limits(self):
        code,n,s=self.run_pair({'plain':True,'memcheck':True})
        self.assertEqual(code,0); self.assertEqual(n,2)
        self.assertFalse(s['complete']); self.assertFalse(s['hardware_cleared']); self.assertFalse(s['historical_failure_resolved'])
    def test_missing_sanitizer_summary_rejects_python_success(self):
        folder=self.root/'memcheck'; folder.mkdir()
        write_json(folder/'summary.json',dict(diagnostic_complete=True,complete=False,image_count=2,
            checkpoint_step=1500,optimizer_updates=0,full_test_evidence=False,integrity_checks={k:True for k in REQUIRED_CHECKS}))
        result=folder/'test_step_001500/result.json';result.parent.mkdir();result.write_text('{}')
        result.with_suffix('.json.sha256').write_text(entry.sha256(result)+'  result.json\n')
        with mock.patch('night_assessment.validate_route_evidence'):
            r=entry.accept(self.root,'memcheck',dict(returncode=0,supervisor_reason=None,sanitizer={'zero_errors_reported':False}),{'names':['a','b']})
        self.assertFalse(r['accepted'])
    def test_extract_all_error_headers(self):
        p=self.root/'sanitizer.log'
        p.write_text('========= COMPUTE-SANITIZER\n========= Program hit cudaErrorMemoryAllocation\n========= Host Frame: ignored\n========= Invalid __global__ write\n========= ERROR SUMMARY: 2 errors\n')
        self.assertEqual(len(entry.headings(p)),3)


if __name__=='__main__': unittest.main()
