import unittest
import torch
from .model import PartitionHead, iterative_assignment, candidate_map, pool_context
from bda_benchmark.experiments.damageformer_joint_region_v1.model import JointHead


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_single_and_grid_match_existing_head(self):
        for new, old in [('single','t3'),('grid','t2')]:
            a=PartitionHead(new,channels=4,dim=8,stride=2)
            b=JointHead(old,channels=4,dim=8,stride=2)
            torch.nn.init.normal_(a.readout[-1].weight,std=.1)
            b.load_state_dict(a.state_dict(),strict=True)
            args=(torch.randn(2,4,5,7),torch.randn(2,4,5,7),torch.rand(2,1,5,7))
            torch.testing.assert_close(a(*args),b(*args),rtol=0,atol=0)

    def test_normalization_borders_empty_groups(self):
        for h,w in [(1,1),(3,5),(7,7)]:
            ids,valid,gh,gw=candidate_map(h,w,2,torch.device('cpu'))
            for bld in [torch.zeros(1,1,h,w),torch.ones(1,1,h,w),torch.rand(1,1,h,w)]:
                q=iterative_assignment(torch.randn(1,9,h,w),ids,valid,bld,gh*gw,2)
                self.assertTrue(torch.isfinite(q).all())
                torch.testing.assert_close(q.sum(1),torch.ones(1,h*w))
                self.assertTrue((q[:,~valid]==0).all())

    def test_full_gradient_after_zero_output_moves(self):
        for variant in ['single','iter1','iter3','grid']:
            head=PartitionHead(variant,channels=4,dim=8,stride=2)
            rgb=torch.randn(2,4,5,7,requires_grad=True)
            feature=torch.randn_like(rgb,requires_grad=True)
            building=torch.rand(2,1,5,7,requires_grad=True)
            self.assertEqual(head(rgb,feature,building).abs().sum().item(),0)
            torch.nn.init.normal_(head.readout[-1].weight,std=.1)
            head(rgb,feature,building).square().mean().backward()
            self.assertIsNone(building.grad)
            self.assertGreater(feature.grad.abs().sum().item(),0)
            if variant!='grid':
                self.assertGreater(head.association[-1].weight.grad.abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_group_isolation_in_assignments_and_pool(self):
        rgb=torch.randn(1,9,4,6)
        building=torch.zeros(1,1,4,6); building[:,:,:,3:]=1
        ids,valid,gh,gw=candidate_map(4,6,2,rgb.device)
        q=iterative_assignment(rgb,ids,valid,building,gh*gw,2)
        changed=rgb.clone(); changed[:,:,:,3:]+=torch.randn_like(changed[:,:,:,3:])*20
        q2=iterative_assignment(changed,ids,valid,building,gh*gw,2)
        torch.testing.assert_close(q.reshape(1,9,4,6)[:,:,:,:3],q2.reshape(1,9,4,6)[:,:,:,:3])
        x=torch.ones(1,1,4,6); x[:,:,:,3:]=100
        torch.testing.assert_close(pool_context(x,q,ids,building,gh*gw),x)

    def test_iter1_iter3_same_parameters_different_assignments(self):
        a=PartitionHead('iter1',channels=4,dim=8,stride=2)
        b=PartitionHead('iter3',channels=4,dim=8,stride=2)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        e=torch.randn(1,9,5,7); building=torch.ones(1,1,5,7)
        ids,v,gh,gw=candidate_map(5,7,2,e.device)
        q1=iterative_assignment(e,ids,v,building,gh*gw,2,1)
        q3=iterative_assignment(e,ids,v,building,gh*gw,2,3)
        self.assertGreater((q1-q3).abs().sum().item(),1e-4)

    def test_cpu_bfloat_autocast(self):
        h=PartitionHead('iter3',channels=4,dim=8,stride=2)
        torch.nn.init.normal_(h.readout[-1].weight,std=.1)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            loss=h(torch.randn(1,4,4,6),torch.randn(1,4,4,6),torch.rand(1,1,4,6)).float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(h.association[-1].weight.grad).all())


if __name__=='__main__': unittest.main()
