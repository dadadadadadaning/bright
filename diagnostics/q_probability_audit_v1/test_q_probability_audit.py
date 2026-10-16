"""Synthetic CPU tests; no BRIGHT data, trained weights, Torch or CUDA."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np

import RUN_Q_PROBABILITY_AUDIT_V1 as audit
audit.init_numpy()
def test_repository():
    if os.environ.get('BRIGHT_REPO'):
        return Path(os.environ['BRIGHT_REPO'])
    for p in (Path(audit.REPO), Path('/workspace/bright')):
        try:
            if (p/'bda_benchmark/experiments/damageformer_srac_q_factorial_v1/compare.py').is_file():
                return p
        except OSError:
            continue
    return Path(audit.REPO)


REPO = test_repository()


class ArchiveFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / 'source'
        self.output = self.root / 'result'
        self.source.mkdir()
        (self.source / 'predictions').mkdir()
        self.gt = np.array([[0,1,1,2],[0,1,3,255]], dtype=np.int16)
        self.predictions = dict(zip(audit.ARMS, [
            np.array([[0,1,0,1],[0,2,3,0]], dtype=np.uint8),
            np.array([[0,1,1,1],[0,1,2,0]], dtype=np.uint8),
            np.array([[1,2,1,2],[0,1,3,0]], dtype=np.uint8)]))
        self.q = np.array([[[.8,.8],[.2,.2]]],dtype=np.float32)
        self.ids = np.array([[0,1],[1,0]], dtype=np.int64)
        self.names = ['event_a_%08d' % i for i in range(450)]
        self.protocol_path = self.root / 'protocol.json'
        audit.write_json(self.protocol_path, dict(selected_images=self.names, arguments=dict(dataset=str(self.root/'dataset'))))
        per_image = {a:audit.confusion(self.gt,p) for a,p in self.predictions.items()}
        # Original compare.checked_eval also checks legacy arms. They are only
        # synthetic gate fixtures, not claims about any real model output.
        all_arms = set(audit.ARMS) | {'a1_pixel','q_region_mean','q_region'}
        self.metric = {}
        for a in all_arms:
            cm = per_image.get(a,per_image[audit.ARMS[0]])*450
            m = audit.metrics(cm);m['valid_miou']=m['foreground_miou']
            self.metric[a]=m
        self.rows=[]
        for i,name in enumerate(self.names):
            file=self.source/'predictions'/('%04d.npz'%i)
            np.savez_compressed(file,target=self.gt,q_new=self.q,region_ids_new=self.ids,
                grid_shape_new=np.array([1,2],dtype=np.int32),
                **{'prediction_'+a:p for a,p in self.predictions.items()})
            self.rows.append(dict(index=i,image=name,event='event_a',
                prediction_file=file.relative_to(self.source).as_posix(),prediction_sha256=audit.digest(file),
                confusion={a:m.tolist() for a,m in per_image.items()}))
        self.report=dict(format='SRAC_Q_FACTORIAL_EVAL_V1',complete=True,full_validation_evidence=True,
            image_count=450,reproduction=dict(passed=True),native_a1_replay_ok=True,test_accessed=False,
            immutable=dict(frozen=True),protocol_sha256=audit.digest(self.protocol_path),
            protocol_path=str(self.protocol_path),source_sha256='a'*64,checkpoint_sha256='b'*64,
            arm='baseline',metrics=self.metric,per_event={'event_a':self.metric})
        self.persist()

    def persist(self):
        audit.write_json(self.source/'result.json',self.report)
        summary=dict(self.report,result_file='result.json',result_sha256=audit.digest(self.source/'result.json'))
        audit.write_json(self.source/'summary.json',summary)
        (self.source/'images.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def args(self, limit=0):
        return SimpleNamespace(source=str(self.source),repo=str(REPO),output=str(self.output),limit=limit,boundary_radius=0)


class NumericTests(unittest.TestCase):
    def test_rescue_harm_fp_and_ignore_counts(self):
        gt=np.array([[0,1,1,2],[0,1,3,255]],dtype=np.int16)
        old=np.array([[0,1,0,1],[0,2,3,3]],dtype=np.uint8)
        new=np.array([[1,2,1,2],[0,1,3,1]],dtype=np.uint8)
        p={audit.ARMS[0]:old,audit.ARMS[1]:old,audit.ARMS[2]:new}
        c=audit.image_counts(gt,p,np.zeros(gt.shape,dtype=np.uint8),np.ones(gt.shape,dtype=np.uint8))['a1_to_q']['all']
        self.assertEqual(int(c.sum()),7)
        s=audit.summarize_cube(c)
        self.assertEqual(s['classes']['Intact']['rescue'],2)
        self.assertEqual(s['classes']['Intact']['harm'],1)
        self.assertEqual(s['classes']['Damaged']['rescue'],1)
        self.assertEqual(s['classes']['Background']['harm'],1)
        np.testing.assert_array_equal(c.sum(2),audit.confusion(gt,old))
        np.testing.assert_array_equal(c.sum(1),audit.confusion(gt,new))
        self.assertEqual(s['classes']['Intact']['fp_added'],1)
        self.assertEqual(s['classes']['Intact']['fp_removed'],1)
        self.assertEqual(s['classes']['Intact']['net_fp'],0)

    def test_building_edges_mark_both_sides_without_class_edges(self):
        gt=np.array([[0,0,1,1,2,2],[0,0,1,1,2,2]],dtype=np.uint8)
        groups=audit.edge_groups(gt,0)
        expected=np.ones(gt.shape,dtype=np.uint8);expected[:,1:3]=0
        np.testing.assert_array_equal(groups,expected)
        self.assertTrue((audit.edge_groups(np.ones((3,3),dtype=np.uint8),0)==1).all())

    def test_ignore_halo_is_separate_from_building_boundary(self):
        gt=np.zeros((7,9),dtype=np.int16);gt[:,4:]=1;gt[3,8]=255
        group=audit.edge_groups(gt,1)
        self.assertEqual(group[3,7],2)
        self.assertEqual(group[0,3],0)
        self.assertEqual(group[0,0],1)

    def test_dilation_matches_manual_square_neighborhood(self):
        a=np.zeros((5,7),dtype=bool);a[0,0]=True;a[3,5]=True
        for radius in (0,1,2):
            expected=np.zeros_like(a)
            for y,x in np.argwhere(a):
                expected[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=True
            np.testing.assert_array_equal(audit.dilate(a,radius),expected)

    def test_q_repeated_ids_and_primary_route_grouping(self):
        gt=np.array([[0,1,1,2],[0,1,3,255]],dtype=np.int16)
        archive=dict(q_new=np.array([[[.8,.8],[.2,.2]]],dtype=np.float32),
                     region_ids_new=np.array([[0,1],[1,0]]),grid_shape_new=np.array([1,2]))
        group,inventory=audit.q_groups(gt,archive)
        np.testing.assert_array_equal(group,np.array([[1,1,2,2],[1,1,2,2]]))
        self.assertAlmostEqual(sum(v['labelled_grid_mass'] for v in inventory.values()),1.75,places=6)
        self.assertEqual(sum(v['regions'] for v in inventory.values()),2)

    def test_q_groups_distinguish_all_four_valid_categories(self):
        gt=np.array([[0,0,1,0,1,2,1,1]],dtype=np.int16)
        archive=dict(q_new=np.ones((1,1,4)),region_ids_new=np.array([[0,1,2,3]]),grid_shape_new=np.array([1,4]))
        groups,_=audit.q_groups(gt,archive)
        np.testing.assert_array_equal(groups,np.array([[0,0,1,1,2,2,3,3]]))

    def test_invalid_routes_and_noninteger_resize_refused(self):
        gt=np.zeros((2,4),dtype=np.int16)
        base=dict(q_new=np.ones((1,1,2)),region_ids_new=np.array([[0,1]]),grid_shape_new=np.array([1,2]))
        for change in ({'q_new':np.zeros((1,1,2))},{'region_ids_new':np.array([[0,2**55]])},
                       {'grid_shape_new':np.array([1,3])},{'region_ids_new':np.array([[0.,1.]])}):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError): audit.q_groups(gt,dict(base,**change))

    def test_invalid_class_ids_are_not_clipped(self):
        with self.assertRaises(ValueError):audit.label_map(np.array([[255]]),'prediction')
        with self.assertRaises(ValueError):audit.label_map(np.array([[-1]]),'target',True)
        with self.assertRaises(ValueError):audit.label_map(np.array([[1.]]),'target',True)

    def test_fixed_class_metric_convention(self):
        cm=np.diag([10,2,3,4]);m=audit.metrics(cm)
        self.assertEqual(m['foreground_miou'],1)
        self.assertEqual(m['oa'],1)
        cm=np.diag([10,0,0,0]);m=audit.metrics(cm)
        self.assertEqual(m['foreground_miou'],0)
        self.assertEqual(m['four_class_miou'],.25)
        self.assertIsNone(m['iou'][1])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.f=ArchiveFixture(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def run_audit(self, limit=0):
        with contextlib.redirect_stdout(io.StringIO()):return audit.run_audit(self.f.args(limit))

    def test_full450_reproduces_source_and_preserves_all_input_bytes(self):
        before={p:audit.digest(p) for p in self.f.source.rglob('*') if p.is_file()}
        r=self.run_audit()
        self.assertTrue(r['diagnostic_complete']);self.assertTrue(r['source_confusions_reproduced'])
        self.assertEqual(r['image_count'],450);self.assertFalse(r['test_accessed'])
        self.assertFalse(r['new_inference']);self.assertNotIn('torch',sys.modules)
        self.assertEqual(before,{p:audit.digest(p) for p in before})
        self.assertEqual(r['pairs']['a1_to_q']['overall']['classes']['Intact']['rescue'],900)
        self.assertEqual(r['pairs']['a1_to_q']['overall']['classes']['Intact']['harm'],450)
        self.assertEqual(len((self.f.output/'images.jsonl').read_text().splitlines()),450)
        self.assertTrue((self.f.output/'groups.csv').is_file());self.assertTrue((self.f.output/'events.csv').is_file())
        self.assertFalse(any(p.suffix in ('.npz','.pth','.pt') for p in self.f.output.rglob('*')))

    def test_limited_run_cannot_claim_full450(self):
        r=self.run_audit(2)
        self.assertTrue(r['complete']);self.assertFalse(r['diagnostic_complete'])
        self.assertFalse(r['full_validation_evidence']);self.assertFalse(r['source_confusions_reproduced'])

    def test_changed_prediction_sha_is_rejected(self):
        p=self.f.source/'predictions/0000.npz';p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Prediction SHA256'):self.run_audit()
        s=audit.read_json(self.f.output/'summary.json');self.assertFalse(s['complete'])

    def test_per_image_confusion_corruption_is_rejected(self):
        self.f.rows[0]['confusion'][audit.ARMS[0]][0][0]+=1;self.f.persist()
        with self.assertRaisesRegex(ValueError,'per-image confusion'):self.run_audit()

    def test_global_confusion_corruption_is_rejected(self):
        self.f.report['metrics'][audit.ARMS[0]]['confusion'][0][0]+=1;self.f.persist()
        with self.assertRaisesRegex(ValueError,'Aggregate confusion'):self.run_audit()

    def test_event_confusion_corruption_is_rejected(self):
        self.f.report['per_event']=copy.deepcopy(self.f.report['per_event'])
        self.f.report['per_event']['event_a'][audit.ARMS[0]]['confusion'][0][0]+=1;self.f.persist()
        with self.assertRaisesRegex(ValueError,'Aggregate confusion'):self.run_audit()

    def test_stored_iou_corruption_is_rejected_after_counts_match(self):
        self.f.report['metrics'][audit.ARMS[0]]['iou'][2]+=.01;self.f.persist()
        with self.assertRaisesRegex(ValueError,'Recomputed metrics'):self.run_audit()

    def test_test_source_rejected_by_original_formal_gate(self):
        self.f.report['test_accessed']=True;self.f.persist()
        with self.assertRaisesRegex(ValueError,'Test evaluation'):self.run_audit()

    def test_changed_source_result_hash_rejected(self):
        p=self.f.source/'result.json';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Source result SHA256'):self.run_audit()

    def test_changed_protocol_hash_rejected(self):
        self.f.protocol_path.write_text(self.f.protocol_path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Bound source protocol'):self.run_audit()

    def test_incomplete_or_reordered_journal_rejected(self):
        self.f.rows[0],self.f.rows[1]=self.f.rows[1],self.f.rows[0];self.f.persist()
        with self.assertRaisesRegex(ValueError,'ordered Val450'):self.run_audit()

    def test_archived_path_escape_rejected(self):
        self.f.rows[0]['prediction_file']='../escape.npz';self.f.persist()
        with self.assertRaisesRegex(ValueError,'Unsafe artifact'):self.run_audit()

    def test_output_cannot_be_inside_source(self):
        args=self.f.args();args.output=str(self.f.source/'new_output')
        with self.assertRaisesRegex(ValueError,'protected source'):
            with contextlib.redirect_stdout(io.StringIO()):audit.run_audit(args)
        self.assertFalse(Path(args.output).exists())

    def test_launcher_parent_rejected_before_writing_any_source_file(self):
        before={p:audit.digest(p) for p in self.f.source.rglob('*') if p.is_file()}
        parent=self.f.source/'forbidden_output'
        with self.assertRaisesRegex(ValueError,'Output parent'):
            audit.main(['--source',str(self.f.source),'--repo',str(REPO),'--output-parent',str(parent)])
        self.assertFalse(parent.exists())
        self.assertEqual(before,{p:audit.digest(p) for p in before})

    def test_cli_background_launcher_and_report(self):
        parent=self.f.root/'outputs'
        cmd=[sys.executable,str(Path(audit.__file__)), '--source',str(self.f.source),'--repo',str(REPO),
             '--output-parent',str(parent),'--limit','2']
        launched=subprocess.run(cmd,text=True,capture_output=True,timeout=15)
        self.assertEqual(launched.returncode,0,launched.stdout+launched.stderr)
        records=list(parent.glob('*.launcher.json'));self.assertEqual(len(records),1)
        record=audit.read_json(records[0]);root=Path(record['output'])
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if (root/'summary.json').is_file():
                state=audit.read_json(root/'summary.json')
                if state.get('complete') or state.get('error'):break
            time.sleep(.1)
        self.assertTrue(state['complete'],Path(record['log']).read_text())
        report=subprocess.run([sys.executable,str(Path(audit.__file__)),'--report',str(root)],text=True,capture_output=True,timeout=15)
        self.assertEqual(report.returncode,0,report.stderr)
        self.assertIn('2/450',report.stdout)


if __name__=='__main__':unittest.main()
