import unittest
import torch
from .audit import metrics, size_masks, accumulate, finalize


class AuditContracts(unittest.TestCase):
    def test_orientation_and_missing(self):
        h=torch.tensor([[8,2,0,0],[1,3,0,0],[0,0,0,0],[0,0,0,0]])
        m=metrics(h)
        self.assertEqual(m['background_to_building'],2)
        self.assertEqual(m['building_to_background'],1)
        self.assertAlmostEqual(m['precision'][1],3/5)
        self.assertAlmostEqual(m['recall'][1],3/4)
        self.assertIsNone(m['iou'][2])

    def test_size_ignore_and_thresholds(self):
        gt=torch.tensor([[1,0,2,2],[0,0,2,2],[255,0,0,0]])
        masks,n=size_masks(gt,2,4)
        self.assertEqual(n,dict(size_small=1,size_medium=0,size_large=1))
        self.assertEqual(int(masks['size_small'].sum()),1)
        self.assertEqual(int(masks['size_large'].sum()),4)

    def test_rescue_harm(self):
        gt=torch.tensor([[0,1,2,3]])
        pred={'A1':torch.tensor([[0,0,2,3]]),
              'S3_split':torch.tensor([[1,1,2,3]]),
              'Grid_split':torch.tensor([[0,1,2,3]])}
        d={}; accumulate(d,pred,gt,torch.ones_like(gt,dtype=torch.bool))
        r=finalize(d)['comparisons']['A1__S3_split']
        self.assertEqual((r['rescued'],r['harmed'],r['net']),(1,1,0))
        self.assertAlmostEqual(r['harm_rate'],1/3)
        self.assertEqual(r['recovery_rate'],1)

    def test_joint_transitions_and_ignore(self):
        gt=torch.tensor([[2,2,2,1,255]])
        pred={'A1':torch.tensor([[2,1,0,1,0]]),
              'S3_split':torch.tensor([[1,2,1,0,0]]),
              'Grid_split':torch.tensor([[2,2,2,1,0]])}
        d={}; accumulate(d,pred,gt,gt!=255,True)
        r=finalize(d)['comparisons']['A1__S3_split']
        self.assertEqual(r['harm_gt_to_wrong'][2][1],1)
        self.assertEqual(r['rescue_gt_from_wrong'][2][1],1)
        self.assertEqual(r['joint_gt_base_other'][2][0][1],1)
        self.assertEqual(r['both_wrong'],1)
        self.assertEqual(r['count'],4)
        empty={}; accumulate(empty,pred,gt,torch.zeros_like(gt,dtype=torch.bool),True)
        e=finalize(empty)['comparisons']['A1__S3_split']
        self.assertIsNone(e['harm_rate'])
        self.assertEqual(sum(map(sum,e['harm_gt_to_wrong'])),0)


if __name__=='__main__': unittest.main()
