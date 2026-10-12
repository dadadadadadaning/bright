"""Strict task initialization, never reinterpret task weights as ImageNet weights."""
import torch
from .common import digest

KNOWN_A1='cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add'

def load_a1(model,path):
    sha=digest(path)
    if sha!=KNOWN_A1:
        raise ValueError('This entry is restricted to the audited A1 SHA256; do not bypass provenance checks.')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('variant')!='a1': raise ValueError('Expected a1 checkpoint')
    # Exact base keys: do not silently discard missing or unexpected parameters.
    model.base.load_state_dict(payload['model'],strict=True)
    return dict(path=str(path),sha256=sha,step=payload['step'],strict=True,
                expected_valid_miou=payload['best_metrics']['valid_miou'])

def optimizer_groups(model,base_lr,new_lr):
    groups=[]
    for name,base,lr in [('base',True,base_lr),('new',False,new_lr)]:
        params=[p for n,p in model.named_parameters()
                if p.requires_grad and n.startswith('base.')==base]
        if not params: raise ValueError('Empty optimizer group '+name)
        groups.append(dict(params=params,lr=lr,initial_lr=lr,name=name))
    return groups
