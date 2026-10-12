import unittest
import torch
from .audit_branches import amplitude,summarize_amplitude,outputs
from .model import JointHead


class FakeBase(torch.nn.Module):
    def forward(self,pre,post,return_aux=False):
        return None,pre,dict(rgb_feature=pre,fused_roi=pre,p_building=post)


class Contracts(unittest.TestCase):
    def test_shift_invariant_amplitude(self):
        b=torch.randn(1,4,3,3)
        r=summarize_amplitude(amplitude(b,torch.ones_like(b)*10,torch.ones(3,3,dtype=torch.bool)))
        self.assertEqual(r['centered_delta_rms'],0)
        self.assertEqual(r['changed'],0)

    def test_intervention_preserves_weights_and_variant(self):
        torch.set_num_threads(1); torch.manual_seed(1)
        m=torch.nn.Module(); m.base=FakeBase(); m.head=JointHead('t3',channels=4,dim=8,stride=2)
        torch.nn.init.normal_(m.head.readout[-1].weight)
        m.eval(); x=torch.randn(1,4,4,4); b=torch.rand(1,1,4,4)
        before={k:v.clone() for k,v in m.state_dict().items()}
        pixel,full,replaced=outputs(m,x,b,True)
        torch.testing.assert_close(pixel,x)
        m.head.variant='t1'
        torch.testing.assert_close(replaced,x+m.head(x,x,b))
        m.head.variant='t3'
        torch.testing.assert_close(full,x+m.head(x,x,b))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        self.assertEqual(m.head.variant,'t3')


if __name__=='__main__': unittest.main()
