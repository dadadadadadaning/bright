import unittest
import torch
from .modules import SplitRegionModule, RegionModule, child_ids, soft_pool_unpool
from bda_benchmark.experiments.damageformer_srac_approx_v1.modules import partition_loss


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_identical_initial_parameters(self):
        torch.manual_seed(1)
        old = RegionModule(channels=4, dim=8)
        torch.manual_seed(1)
        new = SplitRegionModule(channels=4, dim=8)
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)

    def test_single_group_matches_original(self):
        old = RegionModule(channels=4, dim=8, stride=2)
        new = SplitRegionModule(channels=4, dim=8, stride=2)
        new.load_state_dict(old.state_dict())
        rgb, x = torch.randn(2,4,4,4), torch.randn(2,4,4,4)
        for value in (0., 1.):
            building = torch.full((2,1,4,4), value)
            torch.testing.assert_close(old(rgb,x,building)['logits'],
                                       new(rgb,x,building)['logits'])

    def test_pool_and_projection_isolate_groups(self):
        ids = torch.zeros(9,4,dtype=torch.long)
        q = torch.zeros(1,9,4); q[:,4] = 1
        building = torch.tensor([0.,0.,1.,1.])
        idx = child_ids(ids,building)
        x = torch.tensor([1.,3.,100.,200.]).reshape(1,1,1,4)
        result = soft_pool_unpool(x,q,idx,2)['reconstruction'].flatten()
        torch.testing.assert_close(result, torch.tensor([2.,2.,150.,150.]))

    def test_gradients_and_unchanged_partition_loss(self):
        old = RegionModule(channels=4,dim=8,stride=2)
        new = SplitRegionModule(channels=4,dim=8,stride=2)
        new.load_state_dict(old.state_dict())
        rgb, x = torch.randn(2,4,4,4), torch.randn(2,4,4,4)
        building = torch.rand(2,1,4,4,requires_grad=True)
        loc = torch.randint(0,2,(2,4,4))
        a,b = old(rgb,x,building),new(rgb,x,building)
        loss = partition_loss(b,rgb,loc)
        torch.testing.assert_close(loss,partition_loss(a,rgb,loc))
        (b['logits'].square().mean()+.1*loss).backward()
        self.assertIsNone(building.grad)
        for name,p in new.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(new.association[-1].weight.grad.abs().sum().item(),0)

    def test_grid_forward_backward(self):
        model=SplitRegionModule(channels=4,dim=8,stride=2,learned=False)
        x=torch.randn(1,4,4,4)
        output=model(x,x,torch.rand(1,1,4,4))
        output['logits'].square().mean().backward()
        self.assertTrue(torch.isfinite(output['logits']).all())
        self.assertIsNone(model.association[-1].weight.grad)


if __name__ == '__main__':
    unittest.main()
