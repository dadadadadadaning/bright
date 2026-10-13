import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from common import digest, write_json
from gates import find_baselines, inspect_baseline, validate_baseline


class BaselineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.v3 = base / 'SRAC_MEMORY_REBUILD_V3/run_v3'
        (self.v3 / 'keys').mkdir(parents=True)
        (self.v3 / 'keys/paired_bank.pt').write_bytes(b'fixed-parent-bank')
        self.protocol = base / 'protocol.json'
        write_json(self.protocol, {'source': 'synthetic'})
        self.root = base / 'SRAC_REGION_MEMORY_JOINT_V1/run_001'
        self.train = self.root / 'train'
        self.train.mkdir(parents=True)
        ckpt = self.train / 'checkpoint_last.pt'
        ckpt.write_bytes(b'synthetic-serialized-checkpoint-artifact')
        self.bind = {'source_protocol_sha256': digest(self.protocol),
                     'paired_bank_sha256': digest(self.v3 / 'keys/paired_bank.pt')}
        self.metrics = {a: {'foreground_miou': .6, 'iou': [.9,.7,.4,.7]} for a in ('r0','rt','rf')}
        self.meta = dict(format='SRAC_REGION_MEMORY_JOINT_TRAIN_V1', step=5000, mode='train',
                         training_complete=True, sha256=digest(ckpt), protocol_info=self.bind,
                         source_hash='old-code')
        write_json(ckpt.with_suffix('.pt.json'), self.meta)
        write_json(self.train / 'manifest.json', {'source_hash':'old-code','protocol_info':self.bind})
        write_json(self.train / 'input_fingerprints.json', {})
        self.summary = dict(format='SRAC_REGION_MEMORY_JOINT_TRAIN_V1',complete=True,
             full_validation_evidence=True, training_complete=True,smoke=False,test_accessed=False,
             updates_per_arm={a:5000 for a in ('r0','rt','rf')}, metrics=self.metrics,
             integrity={'frozen':True,'bank':True},
             guard_report={'restored':True,'failures':[],'restore_conflicts':[]},
             last_checkpoint=str(ckpt),checkpoint=self.meta,
             input_fingerprints_sha256=digest(self.train / 'input_fingerprints.json'))
        write_json(self.train / 'summary.json',self.summary)
        self.final = dict(image_count=450,full_validation_evidence=True,primary_endpoint=True,
             final_integrity_pending=False,smoke=False,test_accessed=False,
             saved_predictions_reopened=True,final_integrity={'frozen':True},metrics=self.metrics)
        write_json(self.train / 'final_val/result.json',self.final)
        write_json(self.root / 'workflow_state.json',dict(complete=True,full_validation_evidence=True))
        self.args=SimpleNamespace(v3_root=str(self.v3),protocol=str(self.protocol),baseline_root=None)

    def test_auto_discovery_accepts_complete_artifact_set(self):
        value=validate_baseline(self.args)
        self.assertTrue(value['passed'])
        self.assertEqual(value['train_root'],str(self.train))
        self.assertIn(str(self.train/'final_val/result.json'),value['files'])

    def test_explicit_train_or_pipeline_root(self):
        self.args.baseline_root=str(self.train)
        self.assertTrue(validate_baseline(self.args)['passed'])
        self.args.baseline_root=str(self.root)
        self.assertTrue(validate_baseline(self.args)['passed'])

    def test_smoke_and_intermediate_not_accepted(self):
        for change in ({'smoke':True},{'updates_per_arm':{a:1500 for a in ('r0','rt','rf')}},
                       {'complete':False},{'full_validation_evidence':False}):
            write_json(self.train/'summary.json',{**self.summary,**change})
            rows=find_baselines(self.args)
            self.assertFalse(rows[0]['accepted'])
            self.assertTrue(rows[0]['reason'])
            with self.assertRaises(ValueError):validate_baseline(self.args)

    def test_complete_summary_does_not_override_failed_pipeline(self):
        write_json(self.root/'workflow_state.json',{'complete':False,'error':'failed'})
        with self.assertRaisesRegex(ValueError,'pipeline'):
            inspect_baseline(self.args,self.root)

    def test_last_weights_must_match_manifest(self):
        (self.train/'checkpoint_last.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            inspect_baseline(self.args,self.root)

    def test_old_bank_binding_cannot_be_substituted(self):
        (self.v3/'keys/paired_bank.pt').write_bytes(b'different parent bank')
        with self.assertRaisesRegex(ValueError,'different FP32'):
            inspect_baseline(self.args,self.root)

    def test_missing_predictions_reopen_rejected(self):
        write_json(self.train/'final_val/result.json',{**self.final,'saved_predictions_reopened':False})
        with self.assertRaisesRegex(ValueError,'reopened'):
            inspect_baseline(self.args,self.root)

    def test_summary_metric_disagreement_rejected(self):
        other=json.loads(json.dumps(self.summary));other['metrics']['rf']['foreground_miou']=.7
        write_json(self.train/'summary.json',other)
        with self.assertRaisesRegex(ValueError,'metrics disagree'):
            inspect_baseline(self.args,self.root)

    def test_no_candidates_has_actionable_error(self):
        self.args.baseline_root=str(self.root/'missing')
        with self.assertRaisesRegex(ValueError,'find-baseline'):
            validate_baseline(self.args)

    def test_find_does_not_modify_failed_evidence(self):
        write_json(self.train/'summary.json',{**self.summary,'complete':False,'error':'CUDA fault'})
        before={str(p):digest(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertFalse(find_baselines(self.args)[0]['accepted'])
        after={str(p):digest(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)


if __name__=='__main__':unittest.main()
