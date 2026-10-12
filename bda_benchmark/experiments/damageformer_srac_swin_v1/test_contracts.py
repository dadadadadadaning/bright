import unittest
import torch
from .model import SRACSwin, RegionEmbedding, project_probabilities, candidate_map
from .retrieval import SupportClassifier, retrieve
from .build_memory import region_targets

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_projection_and_pool_gradient(self):
        ids,valid,gh,gw=candidate_map(8,8,4,'cpu')
        raw=torch.randn(2,9,64,requires_grad=True)
        q=raw.masked_fill(~valid[None],-1e9).softmax(1)
        f=torch.randn(2,16,8,8,requires_grad=True)
        embedding=RegionEmbedding(16,16)
        z,mass=embedding(f,q,ids,gh*gw)
        p=project_probabilities(z[...,:4],q,ids,(8,8),(16,16))
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(2,16,16),atol=1e-6))
        loss=-p[:,1].clamp_min(1e-8).log().mean()
        loss.backward()
        self.assertGreater(raw.grad.abs().sum().item(),0)
        self.assertGreater(f.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_exclusion_and_empty_fallback(self):
        bank=dict(tokens=torch.randn(6,16),distribution=torch.eye(4)[torch.arange(6)%4],
                  events=torch.tensor([0,0,1,1,2,2]),images=torch.arange(6))
        q=torch.randn(2,16)
        s,d,v=retrieve(q,bank,0,2,6)
        self.assertEqual(v.sum(1).tolist(),[3,3])
        for x in s[v]:
            index=(bank['tokens']==x).all(1).nonzero().item()
            self.assertNotIn(index,[0,1,2])
        head=SupportClassifier(16).eval()
        off=torch.zeros_like(v)
        a=head(q,s,d,off)
        b=head(q,s+100,d.flip(-1),off)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_both_heads_use_all_parameters(self):
        for mode in ('nomemory','memory'):
            head=SupportClassifier(16)
            q=torch.randn(3,16)
            out=head(q,torch.randn(3,5,16),torch.randn(3,5,4).softmax(-1),
                     torch.ones(3,5,dtype=torch.bool),mode)
            out.square().mean().backward()
            for name,p in head.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_label_targets_ignore(self):
        ids,valid,gh,gw=candidate_map(4,4,4,'cpu')
        q=valid.float()[None]; q=q/q.sum(1,keepdim=True)
        r=dict(shape=(4,4),q=q,ids=ids,tokens=torch.zeros(1,gh*gw,16))
        gt=torch.zeros(1,8,8,dtype=torch.long); gt[:,:,4:]=2
        dist,weight=region_targets(gt,r)
        self.assertTrue(torch.allclose(dist[0,0],torch.tensor([.5,0,.5,0])))
        dist,weight=region_targets(torch.full_like(gt,255),r)
        self.assertEqual(weight.sum().item(),0)
        self.assertTrue(torch.isfinite(dist).all())

    def test_full_frontend_forward_backward(self):
        model=SRACSwin(dim=16,stride=4,iterations=3).eval()
        pre=torch.randn(1,3,64,64); post=torch.randn_like(pre)
        r=model(pre,post)
        with torch.no_grad():
            loc,pixel=model.base(pre,post)
        self.assertTrue(torch.allclose(pixel,r['pixel'],atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.allclose(loc,r['loc'],atol=1e-6,rtol=1e-5))
        self.assertEqual(r['prob'].shape,(1,4,64,64))
        self.assertTrue(torch.allclose(r['prob'].sum(1),torch.ones(1,64,64),atol=1e-5))
        loss=-r['prob'][:,2].clamp_min(1e-8).log().mean()
        loss.backward()
        for module in (model.decoder,model.embedding,model.classifier):
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)

    def test_disjoint_optimizer_groups(self):
        from .a1_init import optimizer_groups
        from torch import nn
        model=nn.Module()
        model.base=nn.Linear(4,4)
        model.extra=nn.Linear(4,2)
        groups=optimizer_groups(model,1e-5,1e-4)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertEqual([g['initial_lr'] for g in groups],[1e-5,1e-4])

if __name__=='__main__': unittest.main()
