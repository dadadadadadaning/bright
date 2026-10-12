import unittest
import torch
from .model import JointHead,pool_context


class Contracts(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(7)

    def test_split_isolation(self):
        x=torch.tensor([1.,3.,100.,200.]).reshape(1,1,1,4)
        ids=torch.zeros(9,4,dtype=torch.long)
        q=torch.zeros(1,9,4); q[:,4]=1
        b=torch.tensor([0.,0.,1.,1.]).reshape(1,1,1,4)
        torch.testing.assert_close(pool_context(x,q,ids,b,1).flatten(),torch.tensor([2.,2.,150.,150.]))

    def test_identity_and_gradient_paths(self):
        for v in ('t1','t2','t3','t4'):
            h=JointHead(v,channels=4,dim=8,stride=2)
            x=torch.randn(2,4,4,4,requires_grad=True)
            rgb=torch.randn_like(x,requires_grad=True)
            building=torch.rand(2,1,4,4,requires_grad=True)
            y=h(rgb,x,building)
            self.assertTrue(torch.equal(y,torch.zeros_like(y)))
            # Once zero last layer moves, gradients must reach features/assignment.
            torch.nn.init.normal_(h.readout[-1].weight,std=.1)
            h(rgb,x,building).square().mean().backward()
            self.assertIsNone(building.grad)
            self.assertGreater(x.grad.abs().sum().item(),0)
            if v in ('t3','t4'):
                self.assertGreater(h.association[-1].weight.grad.abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None))

    def test_matched_t3_t4_parameters(self):
        a=JointHead('t3',channels=4,dim=8)
        b=JointHead('t4',channels=4,dim=8)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))


if __name__=='__main__': unittest.main()
