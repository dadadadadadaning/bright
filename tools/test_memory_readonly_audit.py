"""Synthetic tests only: python -m unittest discover -s tools -p test_memory_readonly_audit.py -v"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import memory_readonly_audit as audit


class MemoryAuditTests(unittest.TestCase):
    def bank(self):
        return audit.table(dict(labels=[1,2,2],events=['a','b','b'],images=['i1','i2','i2'],
                                region_ids=[1,2,3],keys=[[1,0],[0,1],[0,1]]))

    def test_event_counts_and_global_cap(self):
        counts,capped,total=audit.SupportCounts(self.bank()).per_class('a','other',True,True,1)
        self.assertEqual(counts,[0,2,0]);self.assertEqual(capped,[0,1,0]);self.assertEqual(total,1)

    def test_same_image_exclusion(self):
        counts,_,_=audit.SupportCounts(self.bank()).per_class('q','i2',False,True,0)
        self.assertEqual(counts,[1,0,0])

    def test_different_regions_not_duplicates(self):
        ids,_=audit.identities(self.bank())
        self.assertEqual(len(set(ids)),3)

    def test_padding_changes_vote(self):
        bank=self.bank();ids,_=audit.identities(bank)
        result=audit.inspect_neighbors([0]+[1]*7,[.9]+[.89]*7,bank,'q','q',True,True,.1,ids)
        self.assertEqual(result['original_vote'],2);self.assertEqual(result['dedup_vote'],1)
        self.assertEqual(result['duplicate_slot_count'],6)

    def test_event_reentry_detected(self):
        result=audit.inspect_neighbors([0,1],[.9,.8],self.bank(),'a','q',True,True,.1)
        self.assertEqual(result['forbidden_active_slots'],1)

    def test_invalid_padding_does_not_vote(self):
        result=audit.inspect_neighbors([0,1],[.9,.99],self.bank(),'q','q',True,True,.1,
                                       declared_valid=[True,False])
        self.assertEqual(result['original_vote'],1)

    def test_no_neighbors_abstains(self):
        self.assertIsNone(audit.vote([],[],np.array([1]),.1))
        idx,_=audit.retrieve(np.array([.9]),np.array([False]),np.array(['a']),8,0,False)
        self.assertEqual(len(idx),0)

    def test_expanded_search_avoids_padding(self):
        score=np.arange(405,0,-1,dtype=float)
        events=np.array(['a']*400+['b','c','d','e','f'])
        allowed=np.ones(405,bool)
        old,_=audit.retrieve(score,allowed,events,8,2,True)
        new,_=audit.retrieve(score,allowed,events,8,2,False)
        self.assertEqual(len(set(old.tolist())),2)
        self.assertEqual(len(set(new.tolist())),7)

    def test_no_identity_is_unknown(self):
        bank=audit.table(dict(labels=[1,1],images=['x','x'],events=['a','a']))
        rows,_=audit.coverage(bank)
        self.assertIsNone(rows[0]['unique_cases'])

    def test_zero_based_labels(self):
        x=audit.table(dict(labels=[-1,0,1,2],images=['a']*4,events=['e']*4),0)
        self.assertEqual(x['labels'].tolist(),[-1,1,2,3])

    def test_test_split_refused(self):
        with self.assertRaises(ValueError): audit.forbid_test({'metadata':{'split':'test'}})
        with self.assertRaises(ValueError): audit.forbid_test({},'test_set.pt')
        audit.forbid_test({'test_accessed':False,'split':'val'},'val_records.pt')

    def test_metrics_abstention(self):
        m=audit.metrics([1,2,2],[1,2,-1])
        self.assertEqual(m['evaluated'],2);self.assertEqual(m['abstained_or_invalid'],1)

    def test_cli_replay_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='bright_memory_audit_') as folder:
            root=Path(folder)
            bank=dict(labels=[1,2],events=['a','b'],images=['a1','b1'],
                      region_ids=[0,0],keys=[[1,0],[.9,.1]])
            query=dict(split='val',labels=[2],events=['c'],images=['c1'],keys=[[1,0]],base_pred=[1])
            for name,obj in [('bank',bank),('query',query)]:
                (root/(name+'.json')).write_text(json.dumps(obj),encoding='utf-8')
            cmd=[sys.executable,str(Path(audit.__file__)), '--memory',str(root/'bank.json'),
                 '--query',str(root/'query.json'),'--output',str(root/'output'),'--replay',
                 '--allow-unverified-key-space','--base-pred-field','base_pred','--exclude-same-event',
                 '--max-per-event','2']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((root/'output/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['damaged_to_intact']['queries'],1)
            self.assertEqual(summary['all']['duplicate_slot_count_affected_fraction'],1.)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
