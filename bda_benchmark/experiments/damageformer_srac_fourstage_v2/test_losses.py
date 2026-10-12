import unittest
import torch
from torch.nn import functional as F
from .losses import region_nll, pixel_ce

class LossLayout(unittest.TestCase):
    def test_private_target_transfer(self):
        from .label_audit import transfer_checked_batch
        image=torch.randn(1,3,4,4)
        loc=torch.zeros(1,4,4)
        damage=torch.ones(1,4,4)
        result=transfer_checked_batch((image,image,loc,damage,['sample']),'cpu')
        loc.fill_(9); damage.fill_(9)
        self.assertEqual(result[2].dtype,torch.long)
        self.assertTrue((result[2]==0).all())
        self.assertTrue((result[3]==1).all())

    def test_ce_value_and_gradient_equivalence(self):
        for classes in (2,4):
            x=torch.randn(2,classes,8,9).contiguous(memory_format=torch.channels_last).requires_grad_()
            ref=x.detach().contiguous().requires_grad_()
            target=torch.randint(0,classes,(2,8,9)); target[:,0]=255
            actual=pixel_ce(x,target)
            expected=F.cross_entropy(ref,target,ignore_index=255)
            actual.backward(); expected.backward()
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(x.grad,ref.grad)

    def test_cpu_label_validation(self):
        from .label_audit import validate_batch_labels
        loc=torch.tensor([[[0,1,255]]])
        damage=torch.tensor([[[0,3,255]]])
        batch=(None,None,loc,damage,['example'])
        self.assertTrue(validate_batch_labels(batch,1,0)['valid'])
        damage[0,0,1]=4
        report=validate_batch_labels(batch,1,0)
        self.assertFalse(report['valid'])
        self.assertEqual(report['samples'][0]['damage']['values'],[0,4,255])
        damage[0,0,1]=3
        loc[0,0,1]=2
        self.assertFalse(validate_batch_labels(batch,1,0)['valid'])

    def check_layout(self, device, dtype=torch.float32):
        torch.manual_seed(42)
        x=torch.randn(2,4,8,9,device=device,dtype=dtype).softmax(1)
        p=x.contiguous(memory_format=torch.channels_last).detach().requires_grad_(True)
        ref=x.float().contiguous().detach().requires_grad_(True)
        target=torch.randint(0,4,(2,8,9),device=device)
        target[:,0,0]=255
        self.assertFalse(p.is_contiguous())
        actual=region_nll(p,target)
        expected=F.nll_loss(ref.clamp_min(1e-8).log(),target,ignore_index=255)
        actual.backward(); expected.backward()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(p.grad,ref.grad.to(dtype))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_cpu_loss_and_gradient_equivalence(self):
        self.check_layout('cpu')

    def test_all_ignore_has_zero_gradient(self):
        p=torch.rand(2,4,8,9,requires_grad=True)
        loss=region_nll(p,torch.full((2,8,9),255))
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertEqual(p.grad.abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_fp32(self):
        self.check_layout('cuda')

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 unavailable')
    def test_cuda_bf16(self):
        self.check_layout('cuda',torch.bfloat16)

if __name__=='__main__': unittest.main()
