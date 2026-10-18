"""CPU adapter contracts. Synthetic metadata/inputs are not BRIGHT evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch

import post_audit as p


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fp(value):
    return hashlib.sha256(json.dumps({k:v.tolist() for k,v in value.items()},sort_keys=True).encode()).hexdigest()


def fixture(root,step=500):
    train=root/'train'
    protocol={'execution_settings':{'cuda_launch_blocking':'1'}}
    config={'chunk_size':2,'seed':3407,'bank_root':'/fixture/parents','values':'/fixture/values'}
    heads={'control':{'weight':torch.tensor([1.,2.])}}
    if step==1500:heads['anchor']={'weight':torch.tensor([1.,3.])}
    saved=dict(format=p.STUDY_FORMAT,step=step,source_mode='fresh',arm='d',test_accessed=False,
               release_sha256=p.STUDY_MANIFEST,runtime_config=config,protocol_info=protocol,
               schedule_horizon=5000,primary_endpoint=1500,heads=heads,input_file_sha256={'/fixture/input':'a'*64})
    checkpoint=train/'checkpoints'/('checkpoint_%06d.pt'%step)
    checkpoint.parent.mkdir(parents=True);torch.save(saved,checkpoint)
    dump(checkpoint.with_suffix('.pt.json'),dict(step=step,sha256=p.digest(checkpoint),head_hashes={k:fp(v) for k,v in heads.items()}))
    base=dict(format=p.STUDY_FORMAT,complete=True,execution_complete=True,training_complete=True,
              full_validation_evidence=True,test_accessed=False)
    dump(root/'workflow_state.json',base)
    dump(train/'summary.json',dict(base,smoke=False,final_integrity_pending=False,source_mode='fresh',arm='d',
                                 endpoint_step=1500,branch_step=500,release_sha256=p.STUDY_MANIFEST))
    manifest=dict(format=p.STUDY_FORMAT,source_mode='fresh',branch_step=500,primary_endpoint=1500,
                  test_accessed=False,arguments={'arm':'d'},train_names=['tr'+str(i) for i in range(2918)],
                  val_names=['va'+str(i) for i in range(450)],release_sha256=p.STUDY_MANIFEST,
                  runtime_config=config,protocol_info=protocol)
    dump(train/'manifest.json',manifest)
    metrics=dict(confusion=np.eye(4,dtype=np.int64).tolist(),iou=[1.,1.,1.,1.],foreground_miou=1.)
    reference=dict(step=step,split='val',image_count=450,whole_val450_processed=True,complete=True,
                   execution_complete=True,final_integrity_pending=False,full_validation_evidence=True,
                   test_accessed=False,metrics={k:copy.deepcopy(metrics) for k in ('grid_probq','control','anchor')})
    path=train/('initial_val' if step==500 else 'val_step_001500')/'result.json';dump(path,reference)
    Path(str(path)+'.sha256').write_text(p.digest(path)+'  result.json\n')
    dump(train/'input_fingerprints.json',saved['input_file_sha256'])
    return saved


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'study';self.saved=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_new_common500_is_bound_without_old_artifacts(self):
        static=p.validate_run(self.root,500)
        self.assertEqual(set(static['meta']['head_hashes']),{'control'})
        self.assertTrue(p.validate_payload(self.saved,static,SimpleNamespace(state_fingerprint=fp),torch))
        p.verify_files(static['files'])
    def test_endpoint_requires_both_new_heads(self):
        root=Path(self.tmp.name)/'endpoint';saved=fixture(root,1500);static=p.validate_run(root,1500)
        self.assertTrue(p.validate_payload(saved,static,SimpleNamespace(state_fingerprint=fp),torch))
        saved['heads'].pop('anchor')
        with self.assertRaisesRegex(ValueError,'head set'):p.validate_payload(saved,static,SimpleNamespace(state_fingerprint=fp),torch)
    def test_incomplete_training_rejected(self):
        path=self.root/'train/summary.json';row=p.read(path);row['complete']=False;dump(path,row)
        with self.assertRaisesRegex(ValueError,'finish training'):p.validate_run(self.root,500)
    def test_smoke_rejected(self):
        path=self.root/'train/summary.json';row=p.read(path);row['smoke']=True;dump(path,row)
        with self.assertRaisesRegex(ValueError,'fresh common500'):p.validate_run(self.root,500)
    def test_wrong_training_arm_rejected(self):
        path=self.root/'train/manifest.json';row=p.read(path);row['arguments']['arm']='b';dump(path,row)
        with self.assertRaisesRegex(ValueError,'fresh common500'):p.validate_run(self.root,500)
    def test_overlapping_names_rejected(self):
        path=self.root/'train/manifest.json';row=p.read(path);row['val_names'][0]=row['train_names'][0];dump(path,row)
        with self.assertRaisesRegex(ValueError,'disjoint'):p.validate_run(self.root,500)
    def test_checkpoint_mutation_rejected(self):
        path=self.root/'train/checkpoints/checkpoint_000500.pt';path.write_bytes(path.read_bytes()+b'mutated')
        with self.assertRaisesRegex(ValueError,'sidecar'):p.validate_run(self.root,500)
    def test_val_checksum_rejected(self):
        path=self.root/'train/initial_val/result.json';row=p.read(path);row['metrics']['control']['iou'][2]=.2;dump(path,row)
        with self.assertRaisesRegex(ValueError,'checksum'):p.validate_run(self.root,500)
    def test_payload_wrong_protocol_rejected(self):
        static=p.validate_run(self.root,500);self.saved['protocol_info']={}
        with self.assertRaisesRegex(ValueError,'provenance'):p.validate_payload(self.saved,static,SimpleNamespace(state_fingerprint=fp),torch)
    def test_payload_wrong_tensor_rejected(self):
        static=p.validate_run(self.root,500);self.saved['heads']['control']['weight'][0]=9
        with self.assertRaisesRegex(ValueError,'tensor fingerprint'):p.validate_payload(self.saved,static,SimpleNamespace(state_fingerprint=fp),torch)
    def test_nonfinite_tensor_rejected(self):
        static=p.validate_run(self.root,500);self.saved['heads']['control']['weight'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):p.validate_payload(self.saved,static,SimpleNamespace(state_fingerprint=fp),torch)
    def test_recorded_input_ledger_must_match(self):
        static=p.validate_run(self.root,500);self.saved['input_file_sha256']['/fixture/input']='b'*64
        with self.assertRaisesRegex(ValueError,'ledger mismatch'):p.validate_payload(self.saved,static,SimpleNamespace(state_fingerprint=fp),torch)
    def test_input_check_uses_completed_ledger(self):
        static=p.validate_run(self.root,500);m=static['manifest']
        runtime=SimpleNamespace(protocol_info=m['protocol_info'],train_names=m['train_names'],val_names=m['val_names'],input_ledger={'/fixture/input':'a'*64})
        p.runtime_input_check(runtime,static)
        runtime.input_ledger['/fixture/input']='wrong'
        with self.assertRaisesRegex(ValueError,'Runtime input'):p.runtime_input_check(runtime,static)
    def test_source_mutation_after_preflight_rejected(self):
        static=p.validate_run(self.root,500)
        path=self.root/'train/summary.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Source changed'):p.verify_files(static['files'])


AUDIT=Path(os.environ.get('BRIGHT_AUDIT_TOOL',os.environ.get('BRIGHT_SLOT_AUDIT_DIR',str(Path(__file__).resolve().parents[1]/'slot_memory_audit_v1_fix1'))))


@unittest.skipUnless((AUDIT/'pixel_stats.py').is_file(),'Install the pinned slot audit for integration tests')
class ReusedStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.package_files(AUDIT,p.AUDIT_MANIFEST)
        cls.pixel=p.load_named(AUDIT/'pixel_stats.py','test_pixel')
        cls.modules=SimpleNamespace(torch=torch,**{n:p.load_named(AUDIT/(f+'.py'),'test_'+n) for n,f in
                            (('slot','slot_stats'),('neighbor','neighbor_stats'),('reader','reader_stats'))})
        # Reuse released synthetic fixtures; never copy the model or algorithms.
        sys.path.insert(0,str(AUDIT))
        cls.fixtures=p.load_named(AUDIT/'test_audit_integration.py','released_fixtures')
    def test_private_pixel_aliases_never_leak_as_old_BD(self):
        acc=p.NamedPixels(self.pixel.PixelAccumulator,1500)
        y=np.array([[1,2,3,0]])
        acc.add(y,dict(grid_probq=y,control=np.array([[2,2,3,0]]),anchor=y),{'all_labeled':np.ones_like(y,dtype=bool)})
        row=acc.report()['all_labeled']
        self.assertEqual(set(row['metrics']),{'grid_probq','control','anchor'})
        self.assertEqual(row['paired']['anchor_vs_control']['rescue_by_gt_class'],[0,1,0,0])
        self.assertNotIn('d_vs_b',json.dumps(row))
    def test_common_head_is_reported_once(self):
        acc=p.NamedPixels(self.pixel.PixelAccumulator,500);y=np.array([[1,2]])
        acc.add(y,dict(grid_probq=y,common500=y),{'all_labeled':np.ones_like(y,dtype=bool)})
        row=acc.report()['all_labeled']
        self.assertEqual(set(row['metrics']),{'grid_probq','common500'})
        self.assertEqual(set(row['paired']),{'common500_vs_grid'})
    def test_replay_uses_new_reference_and_renames(self):
        acc=p.NamedPixels(self.pixel.PixelAccumulator,500);y=np.array([[0,1,2,3]])
        acc.add(y,dict(grid_probq=y,common500=y),{'all_labeled':np.ones_like(y,dtype=bool)})
        reference={'metrics':{k:self.pixel.metrics(np.eye(4,dtype=np.int64)) for k in ('grid_probq','control','anchor')}}
        out=p.replay_check(acc,reference,self.pixel,500)
        self.assertTrue(out['passed']);self.assertIn('common500_vs_initial_anchor',out['paths']);self.assertNotIn('d',out['paths'])
    def test_readers_use_D_support_only(self):
        ex,data=self.fixtures.example_fixture();head=self.fixtures.TinyHead()
        out=p.analyze(ex,data,{'control':head,'anchor':copy.deepcopy(head)},['t0','t1'],['e0','e1'],2,self.modules)
        self.assertEqual(set(out['predictions']),{'grid_probq','control','anchor'})
        self.assertTrue(np.array_equal(out['predictions']['control'],out['predictions']['anchor']))
        self.assertIn('reader_control_damaged_label_mass',out['rows'])
        self.assertIn('local_pred_cosine_vote_damaged',out['rows'])
        self.assertNotIn('reader_b_damaged_label_mass',out['rows'])
    def test_query_GT_only_changes_posthoc_statistics(self):
        ex,data=self.fixtures.example_fixture();heads={'common500':self.fixtures.TinyHead()}
        a=p.analyze(ex,data,heads,['t0','t1'],['e0','e1'],2,self.modules)
        changed=copy.copy(ex);changed.target=torch.tensor([[[3,1]]])
        b=p.analyze(changed,data,heads,['t0','t1'],['e0','e1'],2,self.modules)
        for key in a['predictions']:self.assertTrue(np.array_equal(a['predictions'][key],b['predictions'][key]))
        torch.testing.assert_close(a['rows']['reader_common500_damaged_label_mass'],b['rows']['reader_common500_damaged_label_mass'])
        self.assertFalse(torch.equal(a['slots']['query_labels'],b['slots']['query_labels']))
    def test_run_smoke_reuses_real_stats_and_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'training';saved=fixture(root,1500)
            # Saved trainable payload is the same interface used by the released
            # Reader diagnostic fixture; source and sidecar are real temp files.
            saved['heads']={k:self.fixtures.TinyHead().state_dict() for k in ('control','anchor')}
            cp=root/'train/checkpoints/checkpoint_001500.pt';torch.save(saved,cp)
            dump(cp.with_suffix('.pt.json'),dict(step=1500,sha256=p.digest(cp),head_hashes={k:fp(v) for k,v in saved['heads'].items()}))
            static=p.validate_run(root,1500)
            fixtures=self.fixtures
            class Runtime:
                def __init__(self,args,output):
                    self.device=torch.device('cpu');self.protocol_info=static['manifest']['protocol_info']
                    self.train_names=static['manifest']['train_names'];self.val_names=static['manifest']['val_names']
                    self.input_ledger={'/fixture/input':'a'*64}
                    self.deps=SimpleNamespace(event=lambda n:n)
                    self.bank=dict(images=torch.tensor([0,1]),events=torch.tensor([0,1]),event_names=['e0','e1'])
                    example,_=fixtures.example_fixture()
                    self.values={'rules':{'soft':dict(slot_labels=example.supports['d']['support_labels'],
                                                     slot_valid=example.supports['d']['support_valid'])}}
                def __enter__(self):return self
                def __exit__(self,*args):return False
                def smoke_names(self):return self.train_names[:2],self.val_names[:2]
                def batches(self,names,**kwargs):return iter(names)
                def extract(self,name):return fixtures.example_fixture(name)[1]
                def prepare(self,name):
                    self.extract(name)
                    return fixtures.example_fixture(name)[0]
            def merge(a,b):a['d']['supported_parent_queries']+=b['d']['supported_parent_queries']
            train=SimpleNamespace(JointRegionMemoryClassifier=lambda **kw:fixtures.TinyHead(),
                _formal_runtime=lambda *args:None,_assert_example=lambda *args:None,
                _blank_coverage=lambda:{'d':{'supported_parent_queries':0}},
                _coverage=lambda ex:{'d':{'supported_parent_queries':2}},_merge_coverage=merge,
                _checked_verify=lambda rt:{'frozen':True},_restored_guard=lambda rt:{'restored':True})
            helper=SimpleNamespace(state=SimpleNamespace(state_fingerprint=fp))
            bridge=SimpleNamespace(load_legacy=lambda path:helper,verify_dependencies=lambda helper:True,
                import_runtime=lambda *args:(torch,None,train,SimpleNamespace(verify_release=lambda:True),Runtime))
            original_load=p.load_named
            def load(path,name):return bridge if Path(path).name=='legacy.py' else original_load(path,name)
            args=SimpleNamespace(study_root=str(root),study_dir=str(Path(folder)/'study_package'),audit_dir=str(AUDIT),
                tool_dir=str(Path(folder)/'runtime'),helpers=str(Path(folder)/'helpers'),step=1500,
                output=str(Path(folder)/'diagnostic'),smoke=True)
            with mock.patch.object(p,'package_files',return_value={}),mock.patch.object(p,'load_named',side_effect=load),\
                 mock.patch.object(torch.cuda,'is_available',return_value=True),mock.patch('builtins.print'):
                result=p.run(args)
            self.assertTrue(result['complete']);self.assertEqual(result['image_count'],2)
            self.assertFalse(result['full_validation_evidence']);self.assertFalse(result['test_accessed'])
            self.assertEqual(set(result['pixels']['all_labeled']['metrics']),{'grid_probq','control','anchor'})
            summary=p.read(Path(args.output)/'summary.json')
            self.assertEqual(summary['result_sha256'],p.digest(Path(args.output)/'result.json'))
            self.assertTrue((Path(args.output)/'REPORT.txt').is_file())
            self.assertTrue((Path(args.output)/'regions_summary.csv').is_file())
            self.assertTrue((Path(args.output)/'images.jsonl').is_file())

    def test_invalid_predictions_rejected(self):
        ex,data=self.fixtures.example_fixture();head=self.fixtures.TinyHead();original=head.forward
        def invalid(**kwargs):
            row=original(**kwargs);row['probabilities'][0,0,0,0]=float('nan');return row
        with mock.patch.object(head,'forward',side_effect=invalid):
            with self.assertRaisesRegex(ValueError,'Invalid prediction'):
                p.analyze(ex,data,{'common500':head},['t0','t1'],['e0','e1'],2,self.modules)


if __name__=='__main__':unittest.main()
