import unittest
import torch
from torch import nn
from .audit import BNProbe,probability_paths,parameters_hash
from .train_control import pixel_forward


class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.base=nn.Module()
        for name in ('fusion_layer_1','fusion_layer_3'):
            setattr(self.base,name,nn.Sequential(nn.BatchNorm2d(2),nn.BatchNorm2d(2)))
    def forward(self,x):
        a=self.base.fusion_layer_3(self.base.fusion_layer_1(x))
        b=self.base.fusion_layer_3(self.base.fusion_layer_1(x+10))
        return a,b


class Contracts(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2); torch.manual_seed(42)

    def test_bn_routing_restores_buffers_and_weights(self):
        m=Toy().eval(); probe=BNProbe(m); x=torch.randn(2,2,4,4)
        before={k:v.clone() for k,v in m.state_dict().items()}; h=parameters_hash(m)
        with torch.inference_mode():
            reference=m(x); collected={}
            with probe.route(collect=collected): result=m(x)
            torch.testing.assert_close(reference[0],result[0])
            self.assertEqual(len(collected),8)
            stats=probe.finish(collected,'split')
            a=stats['base.fusion_layer_1.0:0'][0]; b=stats['base.fusion_layer_1.0:1'][0]
            torch.testing.assert_close(b-a,torch.full_like(a,10))
            shared=probe.finish(collected,'shared')
            torch.testing.assert_close(shared['base.fusion_layer_1.0:0'][0],(a+b)/2)
            with probe.route(stats): m(x)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with probe.route(stats):
                    m(x); raise RuntimeError('intentional')
        self.assertEqual(h,parameters_hash(m))
        for k,v in m.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_projection_constant_and_normalized(self):
        from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import candidate_map
        ids,valid,gh,gw=candidate_map(8,8,4,'cpu')
        q=torch.zeros(1,9,64); q[:,4]=1
        logits=torch.tensor([1.,2.,3.,4.])[None,:,None,None].expand(1,4,32,32)
        r=dict(pixel=logits,prob=logits.softmax(1),q=q,ids=ids,shape=(8,8),tokens=torch.zeros(1,gh*gw,4))
        paths=probability_paths(r)
        torch.testing.assert_close(paths['pixel'],paths['projection'])
        for p in paths.values(): torch.testing.assert_close(p.sum(1),torch.ones(1,32,32))

    def test_control_pixel_matches_v2(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        m=SRACSwin().eval(); x=torch.randn(1,3,64,64); y=torch.randn_like(x)
        with torch.inference_mode():
            expected=m(x,y); actual=pixel_forward(m,x,y)
            collected={}; probe=BNProbe(m)
            with probe.route(collect=collected): routed=m(x,y)
            self.assertEqual(len(collected),8)
            torch.testing.assert_close(routed['prob'],expected['prob'])
            with probe.route(probe.finish(collected,'split')): calibrated=m(x,y)
            self.assertTrue(torch.isfinite(calibrated['prob']).all())
        for k in ('loc','pixel'): torch.testing.assert_close(expected[k],actual[k])

    def test_control_losses_backward(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
        from .train_control import ce_lov
        m=SRACSwin().train(); x=torch.randn(2,3,64,64); y=torch.randn_like(x)
        gt=torch.randint(0,4,(2,64,64)); loc=(gt>0).long()
        for mode in ('a1','joint'):
            m.zero_grad(set_to_none=True)
            r=pixel_forward(m,x,y) if mode=='a1' else m(x,y)
            loss=ce_lov(r['loc'],loc)+ce_lov(r['pixel'],gt)*(1 if mode=='a1' else .25)
            if mode=='joint': loss=loss+region_nll(r['prob'],gt)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

if __name__=='__main__': unittest.main()
