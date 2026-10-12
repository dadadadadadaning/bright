"""Four-stage front end: shared RGB Swin, nested decoder, local clustering,
feature-space structural modulation, weighted tokens, full-image projection.
No GT arguments in forward. No pixel-residual fusion in the final prediction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_joint_region_v1.model import JointModel, load_encoder_pretrain
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import iterative_assignment, candidate_map
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool

def block(cin, cout=32):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU())

class NestedDecoder(nn.Module):
    """Small UNet++-style dense skip decoder, not a claimed paper architecture."""
    def __init__(self, channels=(96,192,384,768), width=32):
        super().__init__()
        self.proj=nn.ModuleList([block(c,width) for c in channels])
        self.nodes=nn.ModuleDict({f'{i}_{j}':block((j+1)*width,width)
                                 for j in range(1,4) for i in range(4-j)})
    def forward(self,pyramid):
        x={(i,0):p(f) for i,(p,f) in enumerate(zip(self.proj,pyramid))}
        for j in range(1,4):
            for i in range(4-j):
                up=F.interpolate(x[i+1,j-1],x[i,0].shape[-2:],mode='bilinear',align_corners=False)
                x[i,j]=self.nodes[f'{i}_{j}'](torch.cat([x[i,k] for k in range(j)]+[up],1))
        return x[0,3]

def project_probabilities(logits, q, ids, shape, output_size):
    """Mixture of normalized REGION probabilities, not unnormalized logits."""
    b,k,c=logits.shape
    prob=logits.float().softmax(-1)
    result=q.new_zeros(b,q.shape[-1],c)
    for j in range(9):
        result=result+prob.gather(1,ids[j][None,:,None].expand(b,-1,c))*q[:,j,:,None]
    result=result.transpose(1,2).reshape(b,c,*shape)
    result=F.interpolate(result,output_size,mode='bilinear',align_corners=False)
    return result/result.sum(1,keepdim=True).clamp_min(1e-8)

class RegionEmbedding(nn.Module):
    def __init__(self, channels=256, dim=128):
        super().__init__()
        self.project=nn.Conv2d(channels,dim,1)
        self.score=nn.Conv2d(channels,1,1)
        self.main=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU())
        self.stats=nn.Sequential(nn.LayerNorm(2*channels),nn.Linear(2*channels,dim))
        self.alpha=nn.Parameter(torch.tensor(0.1))
        self.norm=nn.LayerNorm(dim)
    def forward(self,f,q,ids,k):
        weights=self.score(f).float().sigmoid().flatten(2)
        p=soft_pool_unpool(self.project(f).float(),q*weights,ids,k)
        stats=soft_pool_unpool(torch.cat([f.float(),f.float().square()],1),q,ids,k)
        mean,second=stats['tokens'].chunk(2,-1)
        std=(second-mean.square()).clamp_min(0).add(1e-6).sqrt()
        z=self.norm(self.main(p['tokens'])+self.alpha.tanh()*self.stats(torch.cat([mean,std],-1)))
        return z,stats['mass']

class SRACSwin(nn.Module):
    def __init__(self,dim=128,stride=8,iterations=3):
        super().__init__()
        self.config=dict(dim=dim,stride=stride,iterations=iterations)
        self.base=JointModel('t0').base
        self.decoder=NestedDecoder()
        self.embedding=RegionEmbedding(dim=dim)
        self.modulator=nn.Sequential(block(258,64),nn.Conv2d(64,256,1))
        nn.init.zeros_(self.modulator[-1].weight); nn.init.zeros_(self.modulator[-1].bias)
        self.classifier=nn.Linear(dim,4)
    def forward(self,pre,post):
        rp=[self.base._to_nchw(f) for f in self.base.encoder_1(pre)[1:]]
        sp=[self.base._to_nchw(f) for f in self.base.encoder_2(post)[1:]]
        rgb=self.base.fusion_layer_1(self.base._multi_scale_fuse(*rp))
        sar=self.base.fusion_layer_2(self.base._multi_scale_fuse(*sp))
        loc=self.base.clf_1(rgb)
        building=loc.float().softmax(1)[:,1:2]
        fused=self.base.fusion_layer_3(torch.cat([rgb,sar],1))
        pixel=self.base.clf_2(fused*(.1+.9*building.detach()))
        e=self.decoder(rp)
        b,_,h,w=e.shape
        ids,valid,gh,gw=candidate_map(h,w,self.config['stride'],e.device)
        # No additional building hard-split: structure enters modulation once.
        q=iterative_assignment(e,ids,valid,torch.zeros_like(building),gh*gw,
                               self.config['stride'],self.config['iterations'])
        hard=ids[None].expand(b,-1,-1).gather(1,q.argmax(1,keepdim=True)).reshape(b,h,w)
        edge=e.new_zeros(b,1,h,w)
        edge[:,:,1:]=(hard[:,1:]!=hard[:,:-1]).unsqueeze(1)
        edge[:,:,:,1:]=torch.maximum(edge[:,:,:,1:],(hard[:,:,1:]!=hard[:,:,:-1]).unsqueeze(1).to(edge.dtype))
        mod=fused*(.1+.9*building.detach())
        mod=mod+self.modulator(torch.cat([fused,building.detach().to(fused.dtype),edge.detach()],1))
        tokens,mass=self.embedding(mod,q,ids,gh*gw)
        logits=self.classifier(tokens)
        prob=project_probabilities(logits,q,ids,(h,w),pre.shape[-2:])
        return dict(prob=prob,tokens=tokens,mass=mass,q=q,ids=ids,shape=(h,w),
                    loc=F.interpolate(loc,pre.shape[-2:],mode='bilinear',align_corners=False),
                    pixel=F.interpolate(pixel,pre.shape[-2:],mode='bilinear',align_corners=False),
                    entropy=-(q*q.clamp_min(1e-8).log()).sum(1).mean())
