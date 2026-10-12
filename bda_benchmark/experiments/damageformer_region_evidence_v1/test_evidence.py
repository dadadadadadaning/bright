import copy
import unittest
import torch
from .core import extract,intervene,MatchedHead
from .mixing import region_rows


def fixture():
    from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import candidate_map
    ids,_,gh,gw=candidate_map(4,4,2,'cpu')
    q=torch.zeros(1,9,16); q[:,4]=1
    return dict(q=q,ids=ids,shape=(4,4),tokens=torch.zeros(1,gh*gw,4))


class Contracts(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2); torch.manual_seed(42)

    def test_mixing_denominators_ignore_and_components(self):
        r=fixture(); gt=torch.zeros(1,8,8,dtype=torch.long)
        gt[:,0,0]=1; gt[:,3,3]=2; gt[:,7,7]=255
        pixel=gt.clamp(0,3); region=pixel.clone(); region[:,0,0]=0
        rows=region_rows(r,gt,pixel,region)
        self.assertEqual(sum(x['count'] for x in rows),63)
        self.assertAlmostEqual(sum(x['soft_valid_area'] for x in rows),63.,places=4)
        self.assertEqual(sum(x['harm'] for x in rows),1)
        self.assertEqual(rows[0]['hard_gt_foreground_components'],2)
        self.assertEqual(rows[0]['hard_feature_components'],1)
        self.assertGreater(rows[0]['bg_fg_mixing'],0)
        empty=region_rows(r,torch.full_like(gt,255),pixel,region)
        self.assertTrue(all(x['purity'] is None for x in empty))

    def test_matched_heads_constant_features_and_gradients(self):
        r=fixture(); f=torch.randn(1,256,1,1).expand(1,256,4,4)
        pixel=MatchedHead(); region=copy.deepcopy(pixel)
        self.assertEqual(sum(p.numel() for p in pixel.parameters()),sum(p.numel() for p in region.parameters()))
        a=pixel(f,r,'pixel',(8,8)); b=region(f,r,'region',(8,8))
        torch.testing.assert_close(a,b)
        for output,h in ((a,pixel),(b,region)):
            (-output[:,1].clamp_min(1e-8).log().mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))

    def test_real_interventions_fixed_q_and_state(self):
        from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import SRACSwin
        model=SRACSwin().eval(); model.requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}; config=dict(model.config)
        x=torch.randn(1,3,64,64); y=torch.randn_like(x)
        normal=extract(model,x,y)
        from unittest.mock import patch
        with patch.object(model,'forward',side_effect=AssertionError('Must not rerun full frontend')), \
             patch.object(model.decoder,'forward',side_effect=AssertionError('Must not recompute Q')), \
             patch.object(model.base.encoder_2,'forward',side_effect=AssertionError('Must reuse SAR')), \
             patch.object(model.base.clf_1,'forward',side_effect=AssertionError('Must reuse building probability')):
            for mode in ('normal','off','dim_only','edge_only'):
                r=intervene(model,x,normal,mode)
                self.assertIs(r['q'],normal['q'])
                self.assertIs(r['ids'],normal['ids'])
                torch.testing.assert_close(r['pixel'],normal['pixel'],rtol=0,atol=0)
                self.assertEqual(r['region_content'].shape,normal['region_content'].shape)
                if mode=='normal': torch.testing.assert_close(r['prob'],normal['prob'])
        with self.assertRaises(ValueError): extract(model,x,y,'bad')
        self.assertEqual(config,model.config)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        heads=MatchedHead()
        heads(normal['region_content'],normal,'pixel',(64,64))[:,1].mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

if __name__=='__main__': unittest.main()
