"""Observed support metadata only; never infer missing geographic coordinates."""
import torch
from .model import soft_pool_unpool

DISASTERS=('unknown','earthquake','wildfire','flood','volcano','explosion','hurricane','conflict')

def disaster_id(name):
    suffix=str(name).rsplit('-',1)[-1]
    return DISASTERS.index(suffix) if suffix in DISASTERS else 0

def geometry(r):
    h,w=r['shape']; b=r['q'].shape[0]
    y,x=torch.meshgrid(torch.linspace(0,1,h,device=r['q'].device),
                       torch.linspace(0,1,w,device=r['q'].device),indexing='ij')
    value=torch.stack([y,x,y*y,x*x])[None].expand(b,-1,-1,-1)
    pool=soft_pool_unpool(value,r['q'],r['ids'],r['tokens'].shape[1])
    mu=pool['tokens'][...,:2]
    sigma=(pool['tokens'][...,2:]-mu.square()).clamp_min(0).sqrt()
    return torch.cat([pool['mass'][...,None]/(h*w),mu,sigma],-1)
