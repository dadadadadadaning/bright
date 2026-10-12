import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.common import *
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import project_probabilities,modulate_optical
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool


def own_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')): h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def extract(model,pre,post,mode='normal'):
    """Run V2 once and capture all fixed inputs needed by interventions."""
    if mode not in ('normal','off','edge_only','dim_only'): raise ValueError(mode)
    model.eval(); features=[]; sar=[]; loc=[]
    handles=[model.base.fusion_layer_3.register_forward_hook(lambda m,i,o: features.append(o)),
             model.base.fusion_layer_2.register_forward_hook(lambda m,i,o: sar.append(o)),
             model.base.clf_1.register_forward_hook(lambda m,i,o: loc.append(o))]
    try:
        r=model(pre,post)
        if len(features)!=2 or len(sar)!=1 or len(loc)!=1:
            raise RuntimeError('Unexpected V2 feature call counts')
        r['region_content']=features[1].detach()
        r['fixed_sar']=sar[0].detach()
        r['fixed_building']=loc[0].float().softmax(1)[:,1:2].detach()
    finally:
        for handle in handles: handle.remove()
    return r if mode=='normal' else intervene(model,pre,r,mode)


@torch.no_grad()
def intervene(model,pre,reference,mode):
    """Recompute only modulated content/classification using the SAME Q tensor.

    Never invoke full model, decoder, clustering, SAR encoder or building head.
    Avoid comparing independently accumulated CUDA scatter results bit-for-bit.
    """
    if mode not in ('normal','off','edge_only','dim_only'): raise ValueError(mode)
    model.eval()
    q=reference['q']; ids=reference['ids']; h,w=reference['shape']; b=len(pre)
    hard=ids[None].expand(b,-1,-1).gather(1,q.argmax(1,keepdim=True)).reshape(b,h,w)
    edge=q.new_zeros(b,1,h,w)
    edge[:,:,1:]=(hard[:,1:]!=hard[:,:-1]).unsqueeze(1)
    edge[:,:,:,1:]=torch.maximum(edge[:,:,:,1:],(hard[:,:,1:]!=hard[:,:,:-1]).unsqueeze(1).to(edge.dtype))
    floor=1. if mode in ('off','edge_only') else model.config['background_floor']
    strength=0. if mode in ('off','dim_only') else model.config['boundary_strength']
    optical=modulate_optical(pre,reference['fixed_building'],edge,floor,strength)
    mp=[model.base._to_nchw(f) for f in model.base.encoder_1(optical)[1:]]
    rgb=model.base.fusion_layer_1(model.base._multi_scale_fuse(*mp))
    content=model.base.fusion_layer_3(torch.cat([rgb,reference['fixed_sar']],1))
    tokens,mass=model.embedding(content,q,ids,reference['tokens'].shape[1])
    prob=project_probabilities(model.classifier(tokens),q,ids,(h,w),pre.shape[-2:])
    return dict(reference,prob=prob,tokens=tokens,mass=mass,region_content=content)


def load_fixed(path,device):
    sha=digest(path); model,data=load_front(path,device)
    if digest(path)!=sha: raise RuntimeError('Checkpoint changed during load; use fixed snapshot')
    model.requires_grad_(False); model.eval()
    return model,data,sha


def normalized_up(p,size):
    p=F.interpolate(p,size,mode='bilinear',align_corners=False)
    return p/p.sum(1,keepdim=True).clamp_min(1e-8)


class MatchedHead(nn.Module):
    """Identical pointwise classifier for raw pixels and mean region features."""
    def __init__(self,channels=256,dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,dim),nn.GELU(),nn.Linear(dim,4))

    def forward(self,f,r,kind,output_size):
        if kind=='pixel':
            z=self.net(f.permute(0,2,3,1).float()).permute(0,3,1,2)
            return normalized_up(z.softmax(1),output_size)
        if kind!='region': raise ValueError(kind)
        t=soft_pool_unpool(f.float(),r['q'].float(),r['ids'],r['tokens'].shape[1])['tokens']
        return project_probabilities(self.net(t),r['q'].float(),r['ids'],r['shape'],output_size)


def comparisons(gt,base,other):
    ok=(gt>=0)&(gt<4); a=base[ok]==gt[ok]; b=other[ok]==gt[ok]
    return dict(count=int(ok.sum()),base_correct=int(a.sum()),base_wrong=int((~a).sum()),
                rescue=int((~a&b).sum()),harm=int((a&~b).sum()))
