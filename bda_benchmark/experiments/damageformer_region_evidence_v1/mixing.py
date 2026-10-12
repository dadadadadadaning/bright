"""GT is audit-only. Hard connectivity/CC proxies; soft purity uses area GT."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage
from .core import soft_pool_unpool


@torch.no_grad()
def region_rows(r,gt,pixel,region):
    if len(gt)!=1: raise ValueError('Mixing audit requires batch1')
    h,w=r['shape']; k=r['tokens'].shape[1]; q=r['q'].float(); ids=r['ids']
    valid=(gt>=0)&(gt<4)
    safe=gt.clamp(0,3)
    one=F.one_hot(safe.long(),4).permute(0,3,1,2).float()*valid[:,None]
    hist=soft_pool_unpool(F.interpolate(one,(h,w),mode='area'),q,ids,k)
    hist=(hist['tokens']*hist['mass'][...,None])[0].cpu().numpy()
    scale=gt.shape[-2]*gt.shape[-1]/(h*w)
    hard=ids[None].expand(1,-1,-1).gather(1,q.argmax(1,keepdim=True)).reshape(h,w).cpu().numpy()
    # Check connectedness at the actual feature grid, not smoothed display edges.
    pieces=np.zeros(k,dtype=int)
    slices=ndimage.find_objects(hard+1,max_label=k)
    for j,sl in enumerate(slices):
        if sl is not None: pieces[j]=ndimage.label(hard[sl]==j,structure=np.ones((3,3)))[1]
    full=F.interpolate(torch.from_numpy(hard).float()[None,None],gt.shape[-2:],mode='nearest')[0,0].long().numpy()
    target=gt[0].cpu().numpy(); ok=valid[0].cpu().numpy()
    p=pixel[0].cpu().numpy(); pred=region[0].cpu().numpy()
    def counts(mask): return np.bincount(full[mask],minlength=k)
    n=counts(ok); correct=counts(ok&(p==target)); wrong=counts(ok&(p!=target))
    harm=counts(ok&(p==target)&(pred!=target)); rescue=counts(ok&(p!=target)&(pred==target))
    components,nc=ndimage.label((target>0)&(target<4),structure=np.ones((3,3)))
    cross=np.zeros(k,dtype=int)
    if nc:
        pairs=np.unique(np.stack([full[components>0],components[components>0]],1),axis=0)
        cross=np.bincount(pairs[:,0],minlength=k)
    rows=[]
    for j in range(k):
        mass=float(hist[j].sum()); fg=float(hist[j,1:].sum())
        purity=float(hist[j].max()/mass) if mass>1e-8 else None
        rows.append(dict(region_id=j,soft_valid_area=mass*scale,
                         soft_background_area=float(hist[j,0])*scale,soft_intact_area=float(hist[j,1])*scale,
                         soft_damaged_area=float(hist[j,2])*scale,soft_destroyed_area=float(hist[j,3])*scale,
                         purity=purity,bg_fg_mixing=(min(float(hist[j,0]),fg)/mass if mass>1e-8 else None),
                         foreground_class_mixing=(1-float(hist[j,1:].max())/fg if fg>1e-8 else None),
                         hard_feature_components=int(pieces[j]),hard_gt_foreground_components=int(cross[j]),
                         count=int(n[j]),base_correct=int(correct[j]),base_wrong=int(wrong[j]),
                         harm=int(harm[j]),rescue=int(rescue[j]),
                         bucket=('no_valid_mass' if purity is None else 'purity_ge_095' if purity>=.95 else 'purity_080_095' if purity>=.8 else 'purity_lt_080')))
    return rows
