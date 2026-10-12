"""Fixed-space Train-only memory and feature-only neighbor selection."""
import torch
from torch import nn
from torch.nn import functional as F

class SupportClassifier(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.labels=nn.Embedding(4,dim)
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)
        self.norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
        self.out=nn.Linear(dim,4)
    def forward(self,q,support=None,distribution=None,valid=None,mode='memory'):
        if mode=='nomemory':
            # Same learned parameter tensors, active slots but no historical data.
            keys=q[:,None]+self.labels.weight[None]
            values=keys
            valid=torch.ones(keys.shape[:2],dtype=torch.bool,device=q.device)
        else:
            if support is None or distribution is None or valid is None:
                raise ValueError('Support feature/distribution/mask required')
            keys=support
            values=support+distribution @ self.labels.weight
        # Always append a self fallback so empty retrieval is finite.
        keys=torch.cat([keys,q[:,None]],1)
        values=torch.cat([values,q[:,None]],1)
        valid=torch.cat([valid,torch.ones(len(q),1,dtype=torch.bool,device=q.device)],1)
        context,_=self.attn(q[:,None],keys,values,key_padding_mask=~valid,need_weights=False)
        x=self.norm(q+context[:,0])
        return self.out(x+self.ffn(x))

def retrieve(q,bank,event,image,k=8,intervention='normal',generator=None):
    """q has no query labels. Same image AND event excluded conservatively."""
    if k<1 or len(bank['tokens'])<1: raise ValueError('Empty bank or invalid k')
    values=bank['tokens']
    allowed=(bank['events']!=event)&(bank['images']!=image)
    scores=F.normalize(q.float(),dim=-1)@F.normalize(values.float(),dim=-1).T
    if intervention=='random':
        scores=torch.rand(scores.shape,device=q.device,generator=generator)
    scores=scores.masked_fill(~allowed[None],-torch.inf)
    scores,indices=scores.topk(min(k,len(values)),dim=1)
    valid=torch.isfinite(scores)
    support=values[indices]
    dist=bank['distribution'][indices]
    if intervention=='shuffle_labels':
        perm=torch.randperm(len(values),device=q.device,generator=generator)
        dist=bank['distribution'][perm[indices]]
    if intervention=='off': valid=torch.zeros_like(valid)
    return support,dist,valid
