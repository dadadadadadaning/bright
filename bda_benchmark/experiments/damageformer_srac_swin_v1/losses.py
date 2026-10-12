"""Pixel-table losses: equivalent unweighted mean, avoiding NLLLoss2D."""
import torch
from torch.nn import functional as F

def pixel_table(values,target,ignore_index):
    if values.ndim!=4 or target.shape!=(values.shape[0],*values.shape[2:]):
        raise ValueError('Expected BCHW predictions and matching BHW targets')
    # A private label buffer also avoids aliasing another loss's target storage.
    labels=target.long().reshape(-1).clone()
    valid=labels!=ignore_index
    if ((labels[valid]<0)|(labels[valid]>=values.shape[1])).any():
        ids,counts=torch.unique(labels,return_counts=True)
        raise ValueError('GPU target out of range before pixel-table loss: '
                         f'classes={values.shape[1]}, shape={tuple(target.shape)}, '
                         f'values={ids.detach().cpu().tolist()}, counts={counts.detach().cpu().tolist()}')
    table=values.float().permute(0,2,3,1).reshape(-1,values.shape[1]).contiguous()
    return table,labels,valid

def region_nll(probability,target,ignore_index=255):
    table,labels,valid=pixel_table(probability,target,ignore_index)
    if not valid.any(): return table.sum()*0
    # [valid_pixels, classes] dispatches the non-spatial NLL implementation.
    return F.nll_loss(table[valid].clamp_min(1e-8).log().contiguous(),
                      labels[valid].contiguous())

def pixel_ce(logits,target,ignore_index=255):
    table,labels,valid=pixel_table(logits,target,ignore_index)
    if not valid.any(): return table.sum()*0
    return F.cross_entropy(table[valid].contiguous(),labels[valid].contiguous())
