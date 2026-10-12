import unittest
import torch

from .metrics import intact_only_override_cm, metrics_from_cm, fixed_coverage_select
from .regions import local_pair_scores

class ContractTests(unittest.TestCase):
    def test_intact_only_override(self):
        cm=torch.tensor([
            [3,1,0,0],
            [0,4,0,0],
            [0,2,2,0],
            [0,1,0,2],
        ])
        out=intact_only_override_cm(cm)
        self.assertEqual(int(out[:,1].sum()),0)
        self.assertEqual(int(out[:,2].sum()),int(cm[:,2].sum()+cm[:,1].sum()))

    def test_fixed_coverage(self):
        s=torch.arange(100,dtype=torch.float32)
        c=torch.ones(100,dtype=torch.bool)
        sel=fixed_coverage_select(s,c,0.02)
        self.assertEqual(len(sel),2)
        self.assertEqual(sel.tolist(),[99,98])

    def test_local_match_identity(self):
        torch.manual_seed(1)
        q=torch.randn(2,4,8)
        mask=torch.ones(2,4,dtype=torch.bool)
        crit=torch.zeros(2,4)
        logits=torch.zeros(2,4,3)
        m=q[:,None].clone()
        mm=mask[:,None].clone()
        mc=crit[:,None].clone()
        ml=logits[:,None].clone()
        score=local_pair_scores(q,mask,crit,logits,m,mm,mc,ml,weighted=False)
        self.assertTrue(torch.all(score>0.99))

    def test_metrics_finite(self):
        cm=torch.eye(4,dtype=torch.long)*10
        m=metrics_from_cm(cm)
        self.assertAlmostEqual(m["valid_miou"],1.0,places=6)

if __name__=="__main__":
    unittest.main(verbosity=2)
