import unittest
import torch
from .spatial_ops import grid_project, split_project
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map,soft_pool_unpool

class SpatialContracts(unittest.TestCase):
    def setup_case(self):
        ids,valid,gh,gw=candidate_map(4,4,2,torch.device('cpu'))
        q=torch.zeros(1,9,16); q[:,4]=1
        return ids,q,gh*gw
    def test_uniform_mask_unchanged(self):
        ids,q,k=self.setup_case(); x=torch.rand(1,4,4,4)
        original=soft_pool_unpool(x,q,ids,k)['reconstruction']
        for v in (False,True):
            self.assertTrue(torch.allclose(original,split_project(x,q,ids,k,torch.full((1,4,4),v))))
    def test_separates_groups_without_probability_override(self):
        ids,q,k=self.setup_case(); building=torch.zeros(1,4,4,dtype=torch.bool); building[:,:,::2]=True
        x=torch.zeros(1,4,4,4); x[:,0]=(~building).float(); x[:,1]=building.float()
        out=split_project(x,q,ids,k,building)
        self.assertTrue(torch.allclose(out,x))
        self.assertFalse(torch.allclose(grid_project(x,2),x))
    def test_soft_weights_mass_and_batch(self):
        ids,valid,gh,gw=candidate_map(5,7,2,torch.device('cpu'))
        q=torch.randn(2,9,35).masked_fill(~valid[None],-1e4).softmax(1)
        x=torch.rand(2,4,5,7).softmax(1); building=torch.rand(2,5,7)>0.5
        out=split_project(x,q,ids,gh*gw,building)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.sum(1),torch.ones(2,5,7),atol=1e-5))
    def test_stride_one_identity(self):
        x=torch.rand(1,4,5,7)
        self.assertTrue(torch.allclose(grid_project(x,1),x))

if __name__=='__main__': unittest.main()
