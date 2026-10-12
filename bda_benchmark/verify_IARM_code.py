# -*- coding: utf-8 -*-
"""Fast CPU checks for IARM custom modules (does not instantiate Swin)."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from model.DamageFormer_IARM import LightweightFlowAlign, EventBalancedDiagonalGaussianMemory, ResidualEncoder


def main():
    torch.manual_seed(7)
    optical=torch.randn(2,16,8,8)
    sar=torch.randn(2,16,8,8)
    align=LightweightFlowAlign(16,hidden_channels=16,max_offset_pixels=2.0)
    warped,flow=align(optical,sar)
    assert flow.abs().max().item()==0.0, "zero-init flow must start at identity"
    assert torch.allclose(warped,sar,atol=1e-5), "identity grid warp mismatch"

    residual=ResidualEncoder(16)(optical,warped,optical)
    assert residual.shape==(2,16,8,8)

    memory=EventBalancedDiagonalGaussianMemory(embedding_dim=16,num_events=2,momentum=0.9,max_pixels_per_image_class=16)
    embedding=F.normalize(torch.randn(2,16,8,8),dim=1)
    target=torch.zeros(2,32,32,dtype=torch.long)
    target[0,:10]=1; target[0,10:20]=2; target[0,20:]=3
    target[1,:12]=1; target[1,12:24]=2; target[1,24:]=3
    memory.update(embedding,target,torch.tensor([0,1]))
    assert memory.ready(), "all class Gaussians should initialize"
    score=memory.query(embedding)
    assert score.shape==(2,3,8,8)
    variance=memory.variance()
    assert torch.isfinite(score).all() and (variance>0).all()
    print("IARM CUSTOM MODULES: PASS")

if __name__=='__main__': main()
