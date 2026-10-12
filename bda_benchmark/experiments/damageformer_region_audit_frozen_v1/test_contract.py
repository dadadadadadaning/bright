import unittest,numpy as np,torch
from .region_metrics import association_dense_statistics,AuditConfig,audit_one_image
from .instance_utils import canonical_stem,semantic_connected_components
class T(unittest.TestCase):
  def test_stem(self): self.assertEqual(canonical_stem('abc_pre_disaster.tif'),'abc')
  def test_cc(self):
    d=np.zeros((5,5),np.int32); d[:2,:2]=1; d[3:,3:]=2; self.assertEqual(int(semantic_connected_components(d).max()),2)
  def test_mass(self):
    n=4; p=torch.zeros(9,n); p[4]=1; ids=torch.zeros(9,n,dtype=torch.long); ids[4]=torch.tensor([0,0,1,1]); inst=torch.tensor([[1,1],[2,2]]); dam=torch.tensor([[1,1],[2,2]])
    s=association_dense_statistics(p,ids,inst,dam); self.assertAlmostEqual(float(s['total_mass'].sum()),4.0,places=5); b,r=audit_one_image('x',s,inst.numpy(),dam.numpy(),'u',AuditConfig()); self.assertEqual(len(b),2)
if __name__=='__main__': unittest.main()
