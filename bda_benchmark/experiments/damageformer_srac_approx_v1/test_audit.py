import unittest
import torch
from .audit import counts,boundary_mask,metrics
from .modules import soft_pool_unpool,candidate_map

class AuditContracts(unittest.TestCase):
    def test_counts_and_ignore(self):
        gt=torch.tensor([[0,1,2,255]])
        value=counts(torch.tensor([[0,0,2,0]]),torch.tensor([[1,1,2,1]]),gt,gt!=255)
        self.assertEqual((value['rescued'],value['harmed'],value['net']),(1,1,0))
    def test_ignore_is_not_boundary(self):
        self.assertFalse(boundary_mask(torch.tensor([[1,255],[1,255]]),0).any())
    def test_constant_probability_preserved(self):
        ids,_,gh,gw=candidate_map(4,4,2,torch.device('cpu'))
        q=torch.zeros(1,9,16); q[:,4]=1
        x=torch.ones(1,4,4,4)/4
        self.assertTrue(torch.allclose(soft_pool_unpool(x,q,ids,gh*gw)['reconstruction'],x))
    def test_perfect_metrics(self):
        self.assertEqual(metrics(torch.eye(4))['valid_miou'],1)

if __name__=='__main__': unittest.main()
