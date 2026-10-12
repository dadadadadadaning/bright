import torch
from torch import nn
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_approx_v1.modules import candidate_map, masked_local_softmax, normalized_coordinate_grid


def pool_context(x, q, ids, building, count):
    """FP32 batched scatter; no full region-by-pixel matrix or unused reconstruction."""
    b,c,h,w=x.shape
    index=ids[None].expand(b,-1,-1)*2+(building.detach().flatten(2)>=.5).long()
    src=x.float().flatten(2).transpose(1,2)
    mass=x.new_zeros(b,2*count,dtype=torch.float32)
    sums=x.new_zeros(b,2*count,c,dtype=torch.float32)
    for j in range(9):
        mass.scatter_add_(1,index[:,j],q[:,j])
        sums.scatter_add_(1,index[:,j,:,None].expand(-1,-1,c),src*q[:,j,:,None])
    token=sums/mass.clamp_min(1e-6)[:,:,None]
    result=torch.zeros_like(src)
    for j in range(9):
        result=result+token.gather(1,index[:,j,:,None].expand(-1,-1,c))*q[:,j,:,None]
    return result.transpose(1,2).reshape(b,c,h,w)


class JointHead(nn.Module):
    def __init__(self,variant,channels=256,dim=64,stride=8):
        super().__init__()
        self.variant,self.stride=variant,stride
        self.project=nn.Conv2d(channels,dim,1)
        self.association=nn.Sequential(nn.Conv2d(channels+2,64,3,padding=1),nn.GELU(),nn.Conv2d(64,9,1))
        self.readout=nn.Sequential(nn.Conv2d(3*dim,dim,1),nn.GELU(),nn.Conv2d(dim,4,1))
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        if variant in ('t1','t2'): self.association.requires_grad_(False)

    def forward(self,rgb,feature,building):
        z=self.project(feature)
        b,_,h,w=z.shape
        if self.variant=='t1': context=z
        else:
            ids,valid,gh,gw=candidate_map(h,w,self.stride,z.device)
            if self.variant=='t2':
                q=torch.zeros(b,9,h*w,device=z.device,dtype=torch.float32); q[:,4]=1
            else:
                xy=normalized_coordinate_grid(b,h,w,rgb)
                logits=self.association(torch.cat([rgb,xy],1))
                q=masked_local_softmax(logits.float(),valid)
            context=pool_context(z,q,ids,building,gh*gw)
        # Same input width/readout for T1-T4. T4 adds no learnable parameters.
        third=z-context if self.variant=='t4' else z
        return self.readout(torch.cat([z,context.to(z.dtype),third.to(z.dtype)],1))


class JointModel(nn.Module):
    def __init__(self,variant,dim=64,stride=8):
        super().__init__()
        from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
        self.base=DamageFormer_S8_MICB_DSMIL(roi_alpha=.1,mil_residual_scale=0)
        # Unused inherited compatibility modules never enter the optimizer.
        for name,p in self.base.named_parameters():
            if name.startswith(('micb_dsmil.','damage_proto_memory.')): p.requires_grad_(False)
        self.head=None if variant=='t0' else JointHead(variant,dim=dim,stride=stride)

    def forward(self,pre,post):
        loc,pixel,aux=self.base(pre,post,return_aux=True)
        if self.head is None: return loc,pixel
        residual=self.head(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
        return loc,pixel+F.interpolate(residual.float(),pixel.shape[-2:],mode='bilinear',align_corners=False)


def load_encoder_pretrain(model,path):
    """Accept torchvision Swin-T features.* keys only; reject task checkpoints."""
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload.get('state_dict',payload.get('model',payload))
    if not isinstance(state,dict): raise ValueError('Expected Swin-T state_dict')
    state={k.removeprefix('module.'):v for k,v in state.items()}
    expected=model.base.encoder_1.swin_transformer.state_dict()
    supplied={k:v for k,v in state.items() if k.startswith('features.')}
    if set(supplied)!=set(expected):
        raise ValueError('Expected exact torchvision Swin-T features keys. Do not supply A1/B2/VMamba task weights.')
    for encoder in (model.base.encoder_1,model.base.encoder_2):
        encoder.swin_transformer.load_state_dict(supplied,strict=True)
    return dict(feature_tensors=len(supplied),ignored_keys=sorted(set(state)-set(supplied)))
