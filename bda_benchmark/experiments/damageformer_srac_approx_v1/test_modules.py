import unittest
import torch
from .modules import RegionModule, partition_loss, RetrievalClassifier

class Contracts(unittest.TestCase):
    def test_partition_and_gradients(self):
        torch.set_num_threads(2)
        m=RegionModule(channels=8,dim=8,stride=3)
        x=torch.randn(2,8,7,9)
        r=m(x,x,torch.rand(2,1,7,9))
        self.assertTrue(torch.allclose(r['q'].sum(1),torch.ones(2,63),atol=1e-6))
        self.assertEqual(r['logits'].shape,(2,4,7,9))
        loss=r['logits'].square().mean()+partition_loss(r,x,torch.randint(0,2,(2,7,9)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(m.association[-1].weight.grad.abs().sum().item(),0)

    def test_grid_ignores_rgb(self):
        m=RegionModule(channels=8,dim=8,learned=False)
        x=torch.randn(1,8,8,8); b=torch.ones(1,1,8,8)
        self.assertTrue(torch.equal(m(x,x,b)['q'],m(x*0,x,b)['q']))

    def test_retrieval_metadata(self):
        m=RetrievalClassifier(8,3)
        y=m(torch.randn(2,8),torch.randn(2,4,8),torch.randint(0,4,(2,4)),torch.randn(2,4,4),torch.randint(0,3,(2,4)))
        self.assertEqual(y.shape,(2,4))
        y.sum().backward()
        self.assertIsNotNone(m.labels.weight.grad)

if __name__=='__main__': unittest.main()
