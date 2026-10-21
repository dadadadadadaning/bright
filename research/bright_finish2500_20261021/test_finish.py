"""CPU/synthetic contract tests; no BRIGHT/CUDA performance claim."""
import contextlib
import copy
import hashlib
import importlib.util
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
ARCHIVE=HERE/'bright_finish2500_test_v1.pyz'
sys.path.insert(0,str(ARCHIVE))
RESUME=Path(os.environ['BRIGHT_RESUME_TEST_SOURCE'])
ADDON=Path(os.environ['BRIGHT_ABLATION_TEST_SOURCE'])
sys.path.extend([str(RESUME),str(ADDON)])
import torch
import numpy as np
import finish_entry as entry
import train_to_endpoint as training
import assess_endpoint as assessment
import test_endpoint as scoring
from evaluate_checkpoint import ARMS, REQUIRED_CHECKS, write_json
from grid_probe import sha256
from evaluation_observer import EvaluationObserver,track_evaluation
import evidence
import test_resume_integration as fixture_train
import evaluation as original_evaluator
import test_evaluation as fixture_eval
from test_evidence import make_report,write_report


class Training(unittest.TestCase):
    def test_continue_actual_six_arm_adamw_without_evaluation(self):
        f=fixture_train.ResumeIntegrationTests(); f.setUp(); self.addCleanup(f.doCleanups)
        root=f.base/'finish'; root.mkdir()
        with mock.patch.dict(sys.modules), mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(torch.cuda,'is_available',return_value=True), \
             mock.patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            result=training.run_training(f.context(),root)
        s=json.loads((root/'summary.json').read_text())
        self.assertEqual(result,0,s)
        self.assertEqual(s['phase'],'training_complete_evaluation_pending')
        self.assertTrue(s['training_complete']); self.assertFalse(s['complete'])
        self.assertFalse(s['full_validation_evidence']); self.assertFalse(s['test_accessed'])
        self.assertEqual(s['new_updates_per_arm'],{a:4 for a in ARMS})
        self.assertEqual(f.evaluations,[])
        f.assert_checkpoint(root/'checkpoints/checkpoint_001004.pt',1004)
        f.assert_journal(root,1004)
        self.assertEqual(f.snapshot(f.root),f.before)

    def test_partial_paired_failure_keeps_previous_checkpoint(self):
        f=fixture_train.ResumeIntegrationTests(); f.setUp(); self.addCleanup(f.doCleanups)
        root=f.base/'fail'; root.mkdir()
        with mock.patch.dict(sys.modules), mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(torch.cuda,'is_available',return_value=True), \
             mock.patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            code=training.run_training(f.context(fail=True),root)
        s=json.loads((root/'summary.json').read_text())
        self.assertNotEqual(code,0); self.assertEqual(s['last_complete_step'],1002)
        self.assertIn('synthetic B backward failure',s['error'])
        f.assert_checkpoint(root/'checkpoints/checkpoint_001002.pt',1002)
        f.assert_journal(root,1002)
        self.assertEqual(f.evaluations,[])
        self.assertFalse((root/'checkpoints/checkpoint_001004.pt').exists())
        self.assertEqual(f.snapshot(f.root),f.before)


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixture_eval.EvaluationContracts.setUpClass()
    def setUp(self):
        self.f=fixture_eval.EvaluationContracts(); self.f.setUp(); self.addCleanup(self.f.tearDown)
        self.f.heads.update({a:copy.deepcopy(self.f.heads['d']) for a in ('r','l')})
        self.runtime=self.f.runtime()
        self.names=['test%03d' % i for i in range(878)]
        self.state=self.f.legacy.state
        self.output=self.f.folder.parent
        self.observer=EvaluationObserver(self.output,dict(index=-1,image=None,active_arm=None,stage='loading'))
    def run_score(self,batches=None):
        with contextlib.redirect_stdout(io.StringIO()),track_evaluation(self.runtime,self.f.factory,
                original_evaluator,self.names,self.state,self.observer):
            return scoring.evaluate_test(self.runtime,self.f.heads,self.names,self.f.folder,
                seed=3407,original_train=self.f.original,legacy=self.f.legacy,factory=self.f.factory,
                evaluator=original_evaluator,batches=batches or (lambda:self.runtime.batches(self.names,train=False,seed=3407)))
    def test_all878_shared_forward_rng_modes_and_reopened_confusions(self):
        rng=self.state.state_fingerprint(self.state.capture_rng())
        before={a:self.state.state_fingerprint(h.state_dict()) for a,h in self.f.heads.items()}
        modes=self.f.modes(self.runtime)
        with mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No optimizer on Test')):
            result=self.run_score()
        self.assertEqual(result['image_count'],878)
        self.assertTrue(result['test_accessed']); self.assertFalse(result['complete'])
        self.assertFalse(result['full_validation_evidence']); self.assertFalse(result['test_used_for_selection'])
        self.assertEqual(self.runtime.reference_calls,0)
        self.assertEqual(self.runtime.seen,self.names)
        self.assertEqual(len(self.f.factory.forwarded),878*6)
        self.assertEqual(self.f.modes(self.runtime),modes)
        self.assertEqual(self.state.state_fingerprint(self.state.capture_rng()),rng)
        for a,h in self.f.heads.items(): self.assertEqual(self.state.state_fingerprint(h.state_dict()),before[a])
        for pair in result['transitions'].values():
            counts=np.asarray(pair['counts']); a,b=pair['axes'][1:]
            np.testing.assert_array_equal(counts.sum(2),result['metrics'][a]['confusion'])
            np.testing.assert_array_equal(counts.sum(1),result['metrics'][b]['confusion'])
        self.assertEqual(result['journal_sha256'],sha256(self.f.folder/'images.jsonl'))
    def test_wrong_order_stops_without_report(self):
        with self.assertRaisesRegex(ValueError,'order'):
            self.run_score(lambda:iter(reversed(self.names)))
        self.assertTrue(self.observer.rng_restore_skipped)
        self.assertFalse((self.f.folder/'result.json').exists())
    def test_cuda_like_primary_error_preserved_without_rng_cleanup(self):
        error=RuntimeError('synthetic device-side assert')
        def fail(*args,**kwargs): raise error
        self.f.factory.forward=fail
        with mock.patch.object(self.state,'restore_rng',side_effect=AssertionError('Poisoned RNG accessed')):
            with self.assertRaises(RuntimeError) as caught:self.run_score()
        self.assertIs(caught.exception,error)
        self.assertIs(self.observer.first_exception,error)
        self.assertTrue(self.observer.rng_restore_skipped)
        self.assertFalse((self.f.folder/'result.json').exists())
    def test_test_input_cannot_enter_train_scope(self):
        self.names=[self.runtime.train_names[0]]
        with self.assertRaisesRegex(ValueError,'split'):self.run_score()
    def test_query_gt_leak_is_rejected(self):
        self.runtime.leak=True
        with self.assertRaisesRegex(ValueError,'ground-truth'):self.run_score()
    def test_corrupt_transition_is_rejected(self):
        self.names=self.names[:3]
        self.run_score()
        p=self.f.folder/'images.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
        key=next(iter(rows[0]['paired_counts'])); rows[0]['paired_counts'][key][0][0][0]+=1
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'paired'):
            scoring.rebuild(p,self.names,('grid_probq',)+ARMS,original_evaluator.comparison_pairs(ARMS))


class Gates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.names=['val%04d'%i for i in range(450)]
    def make_val(self,split='val'):
        names=self.names if split=='val' else ['train0','train1']
        output=self.root/split; output.mkdir()
        folder=output/('%s_step_002500'%split)
        r=make_report(2500,split,names,ARMS,primary=split=='val')
        write_report(folder,r,names)
        protocol={'execution_settings':{'cuda_launch_blocking':'1'}}
        provenance={'checkpoint_step':2500,'protocol_info':protocol}
        write_json(output/'manifest.json',dict(protocol_info=protocol,names=names))
        return output,folder,r,names,provenance
    def test_finalize_new_val_primary_and_fixed_diagnostic_only(self):
        for split in ('val','fixed_train'):
            out,folder,r,names,p=self.make_val(split)
            final=assessment.finalize(folder/'result.json',r,output=out,names=names,evidence=evidence,
                checks={k:True for k in REQUIRED_CHECKS},provenance=p,primary=split=='val')
            self.assertTrue(final['complete']); self.assertFalse(final['full_test_evidence'])
            self.assertEqual(final['full_validation_evidence'],split=='val')
            self.assertEqual(final['primary_endpoint'],split=='val')
            self.assertEqual(json.loads((folder/'result.pending.json').read_text()),r)
    def test_partial_integrity_rejected(self):
        out,folder,r,names,p=self.make_val()
        checks={k:True for k in REQUIRED_CHECKS}; checks['rng_restored']=False
        with self.assertRaisesRegex(ValueError,'integrity'):
            assessment.finalize(folder/'result.json',r,output=out,names=names,evidence=evidence,
                checks=checks,provenance=p,primary=True)
        self.assertFalse((folder/'result.pending.json').exists())
    def test_test_without_val_not_admitted(self):
        c={'manifest':{'val_names':self.names,'fixed_train_names':['train0']},'parent':{'sha256':'x'}}
        with self.assertRaises(FileNotFoundError):entry.verify_before_test(self.root,c)
    def test_workflow_eval_failure_does_not_restart_training_or_enter_test(self):
        calls=[]
        def run(command,**kwargs):
            stage=command[-1]; calls.append(stage)
            if stage=='train':
                write_json(self.root/'summary.json',dict(last_complete_step=2500,training_complete=True,
                    phase='training_complete_evaluation_pending',new_updates_per_arm={a:500 for a in ARMS}))
                return SimpleNamespace(returncode=0)
            out=self.root/'assessments'/stage; out.mkdir(parents=True)
            write_json(out/'summary.json',dict(complete=False,phase='failed',error='synthetic evaluation failure'))
            return SimpleNamespace(returncode=2)
        with mock.patch.object(entry,'leases',return_value=contextlib.nullcontext()), \
             mock.patch.object(entry,'read_request',return_value={}),mock.patch.object(entry.subprocess,'run',side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            code=entry.coordinate(self.root)
        s=json.loads((self.root/'workflow_summary.json').read_text())
        self.assertEqual(code,2); self.assertTrue(s['training_complete']); self.assertFalse(s['test_complete'])
        self.assertEqual(calls,['train','fixed2000'])
    def test_checkpoint_selection_is_fixed_not_metric_based(self):
        self.assertEqual(entry.STAGES,('train','fixed2000','val2500','fixed2500','test2500'))
        self.assertEqual(assessment.SPECS['test2500'],(2500,'test'))
    def test_build_preserves_inherited_helper_bytes(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            ledger=json.loads(z.read('FINISH_BASE.json'))
            self.assertEqual(ledger['source_sha256'],entry.CERTIFICATE_SHA)
            for n,h in ledger['files'].items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h,n)
            for n in ('finish_entry.py','train_to_endpoint.py','assess_endpoint.py','test_endpoint.py'):
                self.assertEqual(z.read(n),(HERE/n).read_bytes())

class CanonicalTestInputs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        split=self.root/'bda_benchmark/dataset/splitname/standard_ML'; split.mkdir(parents=True)
        self.names=['event_%04d'%i for i in range(878)]
        for key,names in (('train',['train0']),('val',['val0']),('test',self.names)):
            (split/(key+'_set.txt')).write_text('\n'.join(names)+'\n')
        data=self.root/'data'; data.mkdir()
        def paths(dataset,name):
            return {k:Path(dataset)/(name+'_'+k) for k in ('pre','post','label')}
        for name in self.names:
            for p in paths(data,name).values():p.write_bytes(b'synthetic')
        self.runtime=SimpleNamespace(_val_set={'val0'},train_names=['train0'],val_names=['val0'],
            bundle={'args':SimpleNamespace(dataset=str(data))},deps=SimpleNamespace(sample_paths=paths),
            protocol_info={'test_accessed':False},input_ledger={})
        self.manifest={'train_names':['train0'],'val_names':['val0']}
        self.split=split
    def test_explicit_test_scope_without_relabelling_val_or_changing_files(self):
        with contextlib.redirect_stdout(io.StringIO()):
            names,allowed,files=scoring.declare_test(self.runtime,self.manifest,repo=self.root)
        self.assertEqual(names,self.names); self.assertEqual(self.runtime.val_names,['val0'])
        self.assertTrue(self.runtime.protocol_info['test_accessed'])
        self.assertEqual(len(allowed),878*3+3)
        scoring.verify_scope(self.runtime,{},allowed)
        self.runtime.input_ledger[self.root/'undeclared']='x'
        with self.assertRaisesRegex(ValueError,'Undeclared'):
            scoring.verify_scope(self.runtime,{},allowed)
    def test_overlap_and_missing_files_rejected(self):
        p=self.split/'test_set.txt'; p.write_text('train0\n'+'\n'.join(self.names[1:]))
        with self.assertRaisesRegex(ValueError,'disjoint'):
            scoring.declare_test(self.runtime,self.manifest,repo=self.root)
        self.assertFalse(self.runtime.protocol_info['test_accessed'])
        p.write_text('\n'.join(self.names))
        next((self.root/'data').iterdir()).unlink()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileNotFoundError):
            scoring.declare_test(self.runtime,self.manifest,repo=self.root)
        self.assertFalse(self.runtime.protocol_info['test_accessed'])


if __name__=='__main__': unittest.main()
